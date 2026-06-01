from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segment_anything.build_sam import sam_model_registry
from models.encoder import DualStreamEncoder
from models.decoder import MBADecoder
from config import config_dict


class IntermediatePICA(nn.Module):
    def __init__(self, vit_dim: int = 768, prompt_dim: int = 256,
                 attn_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(vit_dim, attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads,
            kdim=prompt_dim, vdim=prompt_dim, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, vit_dim)
        self.norm = nn.LayerNorm(vit_dim)

    def forward(self, vit_tokens: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor) -> torch.Tensor:
        B, H, W, C = vit_tokens.shape
        tokens = vit_tokens.reshape(B, H * W, C)
        q = self.q_proj(tokens)
        attn_out, _ = self.cross_attn(query=q, key=sparse_prompt_embeddings,
                                       value=sparse_prompt_embeddings)
        out = self.out_proj(attn_out)
        tokens = self.norm(tokens + out)
        return tokens.reshape(B, H, W, C)


class SPSPromptEncoder(nn.Module):
    def __init__(self, ori_sam: nn.Module, fix: bool = False):
        super().__init__()
        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            for param in self.sam_prompt_encoder.parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        return self.sam_prompt_encoder(points, boxes, masks)

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()


class DB_SAM(nn.Module):
    mask_threshold: float = 0.0

    def __init__(self, ori_sam: nn.Module = None):
        super().__init__()
        if ori_sam is None:
            ori_sam = sam_model_registry[f'vit_b_{config_dict["img_size"]}'](
                config_dict["checkpoint_path"])
        self.dual_stream_encoder = DualStreamEncoder(ori_sam)
        self._image_encoder = self.dual_stream_encoder.gsa_vit
        self.pica_layers = nn.ModuleList([
            IntermediatePICA(vit_dim=768, prompt_dim=256, num_heads=8)
            for _ in range(4)])
        self.mba_decoder = MBADecoder(ori_sam, self._image_encoder)
        self.sps_prompt_encoder = SPSPromptEncoder(ori_sam=ori_sam, fix=False)

    def _get_sparse_embeddings(self, list_input: List[Dict[str, Any]]) -> torch.Tensor:
        sparse_list = []
        for x in list_input:
            sparse_emb, _ = self.sps_prompt_encoder(
                points=None, boxes=x.get("boxes"), masks=None)
            sparse_list.append(sparse_emb)
        return torch.cat(sparse_list, dim=0)

    def forward(self, list_input: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        sparse_embeddings = self._get_sparse_embeddings(list_input)
        permuted_features, image_embeddings = self.dual_stream_encoder(
            list_input, sparse_embeddings=sparse_embeddings,
            pica_layers=self.pica_layers)
        low_res_masks = self.mba_decoder(
            list_input, permuted_features, image_embeddings, self.sps_prompt_encoder)
        outputs = []
        for low_res_mask in low_res_masks:
            mask = self._postprocess_masks(low_res_mask)
            outputs.append({"masks": mask, "low_res_logits": low_res_mask})
        return outputs

    @torch.no_grad()
    def infer(self, list_input: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        return self.forward(list_input)

    def _postprocess_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            masks,
            (self._image_encoder.img_size, self._image_encoder.img_size),
            mode="bilinear", align_corners=False)

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...],
                          original_size: Tuple[int, ...]) -> torch.Tensor:
        masks = F.interpolate(
            masks, (self._image_encoder.img_size, self._image_encoder.img_size),
            mode="bilinear", align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


def build_db_sam(checkpoint: str = None, load_ori_checkpoint: bool = False) -> DB_SAM:
    sam_builder = sam_model_registry[f'vit_b_{config_dict["img_size"]}']
    if load_ori_checkpoint:
        ori_sam = sam_builder(config_dict['checkpoint_path'])
    else:
        ori_sam = sam_builder(None)
    model = DB_SAM(ori_sam=ori_sam)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in state_dict.items()
                      if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    return model
