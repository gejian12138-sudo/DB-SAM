import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segment_anything.modeling.common import LayerNorm2d
from models.segment_anything.modeling.mask_decoder import MLP


class _SpatialAttention(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class _ChannelAttention(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1), nn.ReLU(),
            nn.Conv2d(in_ch // 4, 4, 1), nn.Softmax(dim=1))

    def forward(self, x):
        return self.gap(self.conv(x))


class _EdgeConv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.diff_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        k = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
        self.diff_conv.weight.data.copy_(k.expand(ch, ch, 3, 3))
        self.diff_conv.requires_grad_(False)

    def forward(self, x):
        return torch.sigmoid(self.diff_conv(x)) * x


class FeatureRecalibrator(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // 8, 1)), nn.ReLU(),
            nn.Linear(max(channel // 8, 1), channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiLevelBoundaryAwareness(nn.Module):
    def __init__(self, in_channels=None, final_dim: int = 32):
        super().__init__()
        in_channels = in_channels or [32, 32, 32, 32]
        self.channel_align = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, final_dim, 1), nn.BatchNorm2d(final_dim), nn.ReLU())
            for c in in_channels
        ])
        self.spatial_attn = _SpatialAttention(final_dim * 4)
        self.channel_attn = _ChannelAttention(final_dim * 4)
        self.feature_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(final_dim * 4, final_dim * 2, 1), nn.ReLU(),
            nn.Conv2d(final_dim * 2, 4, 1), nn.Softmax(dim=1))
        self.edge_refinement = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 3, padding=1),
            nn.BatchNorm2d(final_dim), nn.ReLU(),
            _EdgeConv(final_dim))

    def forward(self, features):
        aligned = []
        for feat, align in zip(features, self.channel_align):
            if feat.shape[-1] != 256:
                feat = F.interpolate(feat, size=256, mode='bilinear', align_corners=False)
            aligned.append(align(feat))
        concat = torch.cat(aligned, dim=1)
        s_attn = self.spatial_attn(concat)
        c_attn = self.channel_attn(concat)
        sel_weights = self.feature_selector(concat)
        fused = sum(aligned[i] * (c_attn[:, i:i + 1] * s_attn * sel_weights[:, i:i + 1])
                    for i in range(4))
        return self.edge_refinement(fused)


class BoundaryAwareFusion(nn.Module):
    def __init__(self, embed_dim: int, transformer_dim: int):
        super().__init__()
        final_dim = 32
        channel_list = [final_dim] * 4
        self.recalibrators = nn.ModuleList([FeatureRecalibrator(c) for c in channel_list])
        self.mba_module = MultiLevelBoundaryAwareness(
            in_channels=channel_list, final_dim=final_dim)
        self.adapters_bridge = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, transformer_dim, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim), nn.GELU(),
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
            ) for _ in range(4)
        ])


class SAMMaskDecoderWrapper(nn.Module):
    def __init__(self, ori_sam: nn.Module, transformer_dim: int = 256):
        super().__init__()
        self.sam_mask_decoder = ori_sam.mask_decoder
        self.med_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4), nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, mba_feature):
        return self._predict_masks(image_embeddings, image_pe,
                                   sparse_prompt_embeddings, dense_prompt_embeddings,
                                   mba_feature)

    def _predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                       dense_prompt_embeddings, mba_feature):
        hq_token_weight = self.med_token.weight
        output_tokens = hq_token_weight.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.sam_mask_decoder.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 0, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_sam = self.sam_mask_decoder.output_upscaling(src)
        upscaled_hq = self.embedding_maskfeature(upscaled_sam) + mba_feature.repeat(b, 1, 1, 1)
        hyper_in = self.hf_mlp(mask_tokens_out).unsqueeze(1)
        b, c, h, w = upscaled_sam.shape
        masks = (hyper_in @ upscaled_hq.view(b, c, h * w)).view(b, -1, h, w)
        return masks


class MBADecoder(nn.Module):
    def __init__(self, ori_sam: nn.Module, image_encoder: nn.Module):
        super().__init__()
        transformer_dim = 256
        embed_dim = image_encoder.embed_dim
        self.mask_decoder = SAMMaskDecoderWrapper(ori_sam=ori_sam)
        self.boundary_aware_fusion = BoundaryAwareFusion(embed_dim, transformer_dim)

    def forward(self, list_input, permuted_features, image_embeddings, prompt_encoder):
        feature_maps_out = [None] * 4
        for i in range(len(permuted_features)):
            projected = self.boundary_aware_fusion.adapters_bridge[4 - i - 1](permuted_features[i])
            feature_maps_out[4 - i - 1] = projected
        feature_maps_out = [
            self.boundary_aware_fusion.recalibrators[i](f)
            for i, f in enumerate(feature_maps_out)]
        mba_feature = self.boundary_aware_fusion.mba_module(feature_maps_out)
        low_res_masks = []
        for image_record, curr_embedding, feat in zip(list_input, image_embeddings, mba_feature):
            points = None
            if image_record.get("point_coords") is not None:
                points = (image_record["point_coords"], image_record["point_labels"])
            sparse_embeddings, dense_embeddings = prompt_encoder(
                points=points, boxes=image_record.get("boxes"),
                masks=image_record.get("mask_inputs"))
            low_res_mask = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                mba_feature=feat.unsqueeze(0))
            low_res_masks.append(low_res_mask)
        return low_res_masks
