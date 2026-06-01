import math
import torch
import torch.nn as nn

from models.segment_anything.modeling.common import LayerNorm2d, Reshaper
from config import config_dict
from utils.utils import preprocess


# ===========================================================================
# GSA: Global Semantic Adaptation (LoRA + Trainable Normalization)
# ===========================================================================

class LoRA_qkv(nn.Module):
    def __init__(self, qkv_layer: nn.Linear, rank: int, scale: float = 1.0):
        super().__init__()
        self.original_qkv = qkv_layer
        self.dim = qkv_layer.in_features
        self.rank = rank
        self.scale = scale
        self.lora_A = nn.Linear(self.dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, 3 * self.dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        orig_out = self.original_qkv(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scale
        return orig_out + lora_out[:, :, :3 * self.dim]


class LoRABlock(nn.Module):
    def __init__(self, original_block: nn.Module, rank: int = 8):
        super().__init__()
        self.block = original_block
        self.block.attn.qkv = LoRA_qkv(self.block.attn.qkv, rank)
        for param in self.block.parameters():
            param.requires_grad = False
        for param in self.block.attn.qkv.lora_A.parameters():
            param.requires_grad = True
        for param in self.block.attn.qkv.lora_B.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.block(x)


class GSAViT(nn.Module):
    def __init__(self, ori_sam: nn.Module, fix: bool = True, lora_rank: int = 8):
        super().__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        self.sam_img_encoder.blocks = nn.ModuleList([
            LoRABlock(blk, rank=lora_rank) for blk in self.sam_img_encoder.blocks
        ])
        self.patch_size = self.sam_img_encoder.patch_size
        self.depth = self.sam_img_encoder.depth
        self.embed_dim = self.sam_img_encoder.embed_dim
        self.img_size = self.sam_img_encoder.img_size
        self.global_index = self.sam_img_encoder.global_index
        if fix:
            for name, param in self.sam_img_encoder.named_parameters():
                if 'lora_A' not in name and 'lora_B' not in name:
                    param.requires_grad = False
            for name, module in self.sam_img_encoder.named_modules():
                if 'block' in name and isinstance(module, (nn.LayerNorm, LayerNorm2d)):
                    for param in module.parameters():
                        param.requires_grad = True
            for module in self.sam_img_encoder.neck.modules():
                if isinstance(module, (nn.LayerNorm, LayerNorm2d)):
                    for param in module.parameters():
                        param.requires_grad = True

    def forward_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
        return x

    def forward_block(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.sam_img_encoder.blocks[idx](x)

    def forward_neck(self, x: torch.Tensor) -> torch.Tensor:
        return self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))


# ===========================================================================
# SCA: Spatial-Channel Attention (MDP feature calibration)
# ===========================================================================

class _ChannelGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid), nn.ReLU(inplace=True), nn.Linear(mid, in_channels))

    def forward(self, x):
        b, c, _, _ = x.size()
        return self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)


class _SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.conv(torch.cat([avg_out, max_out], dim=1))


class SpatialChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_gate = _ChannelGate(in_channels, reduction)
        self.spatial_gate = _SpatialGate()

    def forward(self, x):
        return x * torch.sigmoid(self.channel_gate(x) + self.spatial_gate(x))


# ===========================================================================
# MDP: Multi-Scale Detail Perception
# ===========================================================================

class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
        return x / keep_prob * random_tensor


def get_msc_model_config(scale='tiny'):
    configs = {
        'tiny': {'embed_dims': [32, 64, 160, 256], 'depths': [3, 3, 3, 3], 'drop_path_rate': 0.1},
    }
    cfg = configs[scale]
    return cfg['embed_dims'], cfg['depths'], cfg['drop_path_rate']


class _DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = _DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2), nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x, x.flatten(2).transpose(1, 2), x.shape[2], x.shape[3]


class _AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = (self.conv0_2(self.conv0_1(attn))
                + self.conv1_2(self.conv1_1(attn))
                + self.conv2_2(self.conv2_1(attn)))
        return self.conv3(attn) * u


class _MDP_SA(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = _AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class _MDPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = _MDP_SA(dim)
        self.drop_path = _DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = _Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x.view(B, C, N).permute(0, 2, 1)


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        return self.norm(x).flatten(2).transpose(1, 2), H, W


def _trunc_normal_init(m: nn.Module, std: float = 0.02) -> None:
    nn.init.trunc_normal_(m.weight, std=std)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def _constant_init(m: nn.Module, val: float) -> None:
    nn.init.constant_(m.weight, val)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def _normal_init(m: nn.Module, mean: float = 0., std: float = 1.) -> None:
    nn.init.normal_(m.weight, mean, std)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


class MDPNet(nn.Module):
    def __init__(self, in_chans=3, embed_dims=None, mlp_ratios=None,
                 drop_rate=0., drop_path_rate=0.1, depths=None, num_stages=4):
        super().__init__()
        embed_dims = embed_dims or [32, 64, 160, 256]
        mlp_ratios = mlp_ratios or [8, 8, 4, 4]
        depths = depths or [3, 3, 3, 3]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = _StemConv(3, embed_dims[0])
            else:
                patch_embed = _OverlapPatchEmbed(
                    patch_size=3, stride=2, in_chans=embed_dims[i - 1], embed_dim=embed_dims[i])
            block = nn.ModuleList([_MDPBlock(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _trunc_normal_init(m, std=.02)
            elif isinstance(m, nn.LayerNorm):
                _constant_init(m, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                _normal_init(m, 0, math.sqrt(2.0 / fan_out))

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(4):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            if i == 0:
                x, H, W = patch_embed(x)[1:]
            else:
                x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class MDPBranch(nn.Module):
    def __init__(self, scale='tiny', pretrained=None):
        super().__init__()
        embed_dims, depths, drop_path_rate = get_msc_model_config(scale)
        self.model = MDPNet(embed_dims=embed_dims, depths=depths,
                            drop_path_rate=drop_path_rate)
        self.channel_list = embed_dims

    def forward(self, x):
        return self.model(x)


# ===========================================================================
# DCA: Dual-stream Collaborative Attention
# ===========================================================================

class DCAModule(nn.Module):
    def __init__(self, channel_list, embed_dim, patch_size, global_index):
        super().__init__()
        img_size = config_dict["img_size"]
        num_patches = img_size // patch_size
        self.sca_modules = nn.ModuleList([
            SpatialChannelAttention(in_channels=channel_list[i])
            for i in range(len(global_index))
        ])
        self.attn_unit = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=1, groups=embed_dim // 8),
            nn.Conv2d(embed_dim * 2, embed_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, 2, 1),
            nn.Softmax(dim=1),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1),
            nn.Sigmoid(),
        )
        self.adapters_in = nn.ModuleList([
            Reshaper(size_in=img_size // 4, channel_in=channel_list[0],
                     size_out=num_patches, channel_out=embed_dim),
            Reshaper(size_in=img_size // 8, channel_in=channel_list[1],
                     size_out=num_patches, channel_out=embed_dim),
            Reshaper(size_in=img_size // 16, channel_in=channel_list[2],
                     size_out=num_patches, channel_out=embed_dim),
            Reshaper(size_in=img_size // 32, channel_in=channel_list[3],
                     size_out=num_patches, channel_out=embed_dim),
        ])


# ===========================================================================
# Dual-Stream Encoder (GSA + MDP + DCA)
# ===========================================================================

class DualStreamEncoder(nn.Module):
    def __init__(self, ori_sam: nn.Module):
        super().__init__()
        self.gsa_vit = GSAViT(ori_sam=ori_sam, fix=True)
        self.mdp_branch = MDPBranch(scale='tiny', pretrained=None)
        channel_list = self.mdp_branch.channel_list
        embed_dim = self.gsa_vit.embed_dim
        patch_size = self.gsa_vit.patch_size
        self.global_index = self.gsa_vit.global_index
        self.dca = DCAModule(channel_list, embed_dim, patch_size, self.global_index)

    def forward(self, list_input, sparse_embeddings=None, pica_layers=None):
        input_images = torch.stack([
            preprocess(x["image"], pixel_mean=config_dict["pixel_mean"],
                       pixel_std=config_dict["pixel_std"])
            for x in list_input
        ], dim=0)
        vit_tokens = self.gsa_vit.forward_patch_embed(input_images)
        mdp_features = self.mdp_branch(input_images)
        permuted_features = [None] * 4

        for i in range(len(self.global_index)):
            for j in range(2):
                vit_tokens = self.gsa_vit.forward_block(vit_tokens, i * 3 + j)
            mdp_features[i] = self.dca.sca_modules[i](mdp_features[i])
            mdp_adapted = self.dca.adapters_in[i](mdp_features[i])
            vit_4d = vit_tokens.permute(0, 3, 1, 2)
            mdp_4d = mdp_adapted
            concat = torch.cat([vit_4d, mdp_4d], dim=1)
            attn_weights = self.dca.attn_unit(concat)
            attn_weights = self.dca.spatial_attn(attn_weights)
            w_vit = attn_weights[:, 0:1, :, :]
            w_mdp = attn_weights[:, 1:2, :, :]
            fused = vit_4d * w_vit + mdp_4d * w_mdp
            vit_tokens = fused.permute(0, 2, 3, 1)
            vit_tokens = self.gsa_vit.forward_block(vit_tokens, self.global_index[i])
            if pica_layers is not None and sparse_embeddings is not None:
                vit_tokens = pica_layers[i](vit_tokens, sparse_embeddings)
            permuted_features[i] = vit_tokens.permute(0, 3, 1, 2)

        image_embeddings = self.gsa_vit.forward_neck(vit_tokens)
        return permuted_features, image_embeddings
