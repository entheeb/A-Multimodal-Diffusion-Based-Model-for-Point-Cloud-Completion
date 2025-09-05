# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MCC: https://github.com/facebookresearch/MCC
# Point-E: https://github.com/openai/point-e
# RIN: https://arxiv.org/pdf/2212.11972
# This code includes the implementation of our default two-stream model.
# Our default two-stream implementation is based on RIN and MCC,
# Other backbone in the two-stream family such as PerceiverIO will also work.
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
#import torch.nn.functional as F

from functools import partial
#from timm.models.vision_transformer import PatchEmbed, Block
#from .util import get_2d_sincos_pos_embed, preprocess_img
from .modules import Denoiser_backbone

'''class XYZPosEmbed(nn.Module):
    """
    A Masked Autoencoder with VisionTransformer backbone.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim

        self.two_d_pos_embed = nn.Parameter(
            torch.zeros(1, 64 + 1, embed_dim), requires_grad=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.win_size = 8

        self.pos_embed = nn.Linear(3, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(1)
        ])

        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim,))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], 8, cls_token=True)
        self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.invalid_xyz_token, std=.02)

    def forward(self, seen_xyz, valid_seen_xyz):
        emb = self.pos_embed(seen_xyz)

        emb[~valid_seen_xyz] = 0.0
        emb[~valid_seen_xyz] += self.invalid_xyz_token

        B, H, W, C = emb.shape
        emb = emb.view(B, H // self.win_size, self.win_size, W // self.win_size, self.win_size, C)
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.win_size * self.win_size, C)

        emb = emb + self.two_d_pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.two_d_pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(emb.shape[0], -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)
        for _, blk in enumerate(self.blocks):
            emb = blk(emb)
        return emb[:, 0].view(B, (H // self.win_size) * (W // self.win_size), -1)

class MCCEncoder(nn.Module):
    """ 
    MCC's RGB and XYZ encoder
    """
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, 
                 num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, drop_path=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.n_tokens = num_patches + 1
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.cls_token_xyz = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.xyz_pos_embed = XYZPosEmbed(embed_dim, num_heads)

        self.blocks_xyz = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for i in range(depth)])

        self.norm_xyz = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_token_xyz, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, seen_xyz, valid_seen_xyz):

        # get tokens
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        y = self.xyz_pos_embed(seen_xyz, valid_seen_xyz)

        ##### forward E_XYZ #####
        # append cls token
        cls_token_xyz = self.cls_token_xyz
        cls_tokens_xyz = cls_token_xyz.expand(y.shape[0], -1, -1)

        y = torch.cat((cls_tokens_xyz, y), dim=1)
        # apply Transformer blocks
        for blk in self.blocks_xyz:
            y = blk(y)
        y = self.norm_xyz(y)

        ##### forward E_RGB #####
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # combine encodings
        return torch.cat([x, y], dim=2)'''
    
# ---------- Fourier position encoding on XYZ ----------
class FourierPE(nn.Module):
    def __init__(self, num_freqs: int = 8, scale: float = 0.5):
        """
        num_freqs  – number of sine/cosine bands per coordinate (total PE dim = 3*2*num_freqs)
        scale      – normalise XYZ to roughly [-scale, scale] before encoding
        """
        super().__init__()
        freqs = 2. ** torch.arange(num_freqs) * math.pi / scale   # [F]
        self.register_buffer("freqs", freqs)

    def forward(self, xyz):               # xyz: [B, N, 3]
        # [B, N, 3, F]
        enc = xyz.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1) # [B, N, 3, 2F]
        return enc.flatten(-2)  
    
    
def build_2d_sincos_position_embedding(h, w, dim, temperature=10_000.0):
    """
    h × w  – patch grid size (e.g. 16×16 for 32-pixel patches on 512×512 depth)
    dim    – embedding dimension (must be even)
    Returns a (h·w, dim) tensor ready to add to patch embeddings.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2-D sine-cosine PE"

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    y = y.reshape(-1).float()     # [h*w]
    x = x.reshape(-1).float()

    div = torch.exp(
        torch.arange(0, dim // 2, 2).float() * -(math.log(temperature) / (dim // 4))
    )                              # [dim/4]

    pe = torch.zeros(h * w, dim)
    pe[:, 0:dim//4]           = torch.sin(x[:, None] * div)   # cos/sin on x
    pe[:, dim//4:dim//2]      = torch.cos(x[:, None] * div)
    pe[:, dim//2:3*dim//4]    = torch.sin(y[:, None] * div)   # cos/sin on y
    pe[:, 3*dim//4:dim]       = torch.cos(y[:, None] * div)
    return pe                # (h·w, dim)                                  # [B, N, 6F]

# --- Condition Embedding Modules ---

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, class_labels):
        x = self.embedding(class_labels)
        return self.norm(x).unsqueeze(1)  # [B, 1, embed_dim]


class ViewAngleEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)) 

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, view_angles):
        return self.mlp(view_angles).unsqueeze(1)  # [B, 1, embed_dim]


class PartialPointCloudEncoder(nn.Module):

    def __init__(self, input_dim: int = 3, embed_dim: int = 256,
                 num_tokens: int = 256, num_layers: int = 8,
                 num_heads: int = 8, num_freqs: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.input_proj = nn.Linear(input_dim, embed_dim)  # [B,N,embed_dim]

        # encoder over all 1024 points
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                               dim_feedforward=embed_dim * 4,
                                               batch_first=True, dropout=0.1, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # global CLS token & learned queries
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens -1, embed_dim))

        # decoder + refiner
        dec_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                               dim_feedforward=embed_dim * 4,
                                               batch_first=True, dropout=0.1, activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers // 2)

        query_refiner_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                         nhead=num_heads,
                                                         dim_feedforward=embed_dim * 4,
                                                         batch_first=True,
                                                         dropout=0.1, activation="gelu", norm_first=True)
        self.query_refiner = nn.TransformerEncoder(query_refiner_layer,
                                                   num_layers=num_layers // 2)

        self.ln_out = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.token_queries)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, pcd):
        B, N, _ = pcd.shape
        x = self.input_proj(pcd)                   # [B,N,D]

        # prepend CLS
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)          # [B,1+N,D]

        x = self.encoder(x)
        patch_tokens = x[:, 1:, :]                  # drop CLS for decoder
        cls_out = x[:, 0:1, :]

        # learned queries
        q = self.token_queries.expand(B, -1, -1)   # [B,T-1,D]
        tokens = self.decoder(q, patch_tokens)     # [B,T-1,D]
        tokens = tokens + self.query_refiner(tokens)

        tokens = torch.cat([cls_out, tokens], dim=1)  # [B,T,D]
        return self.ln_out(self.proj_out(tokens))


class DepthMapEncoder(nn.Module):

    def __init__(self, in_channels: int = 1, embed_dim: int = 256,
                 num_tokens: int = 64, patch: int = 32, num_layers: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.patch = patch

        # patch projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)

        # fixed 2‑D sin‑cos positional encoding
        h = w = 512 // patch
        pe_mat = build_2d_sincos_position_embedding(h, w, embed_dim)
        self.register_buffer("pos_embed", pe_mat)

        # Transformer mixer over patch tokens (+ CLS)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
                                               batch_first=True, dropout=0.1, activation="gelu", norm_first=True)
        self.mixer = nn.TransformerEncoder(enc_layer, num_layers= num_layers)

        # global CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        #learned query tokens
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens -1, embed_dim))

        # decoder to extract latent geometry tokens
        dec_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
                                               batch_first=True, dropout=0.1, activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers= num_layers // 2)

        query_refiner_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
                                                         batch_first=True,
                                                         dropout=0.1, activation="gelu", norm_first=True)
        self.query_refiner = nn.TransformerEncoder(query_refiner_layer,
                                                   num_layers=num_layers // 2)

        self.ln_out = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.token_queries)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, depth_maps):
        x = self.proj(depth_maps)                 # [B, D, H, W]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, HW, D]

        x = x + self.pos_embed[None, :, :]        # add fixed PE

        # prepend CLS
        cls_tok = self.cls_token.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([cls_tok, x], dim=1)          # [B, 1+HW, D]

        x = self.mixer(x)                          # CLS updated

        patch_tokens = x[:, 1:, :]                 # discard CLS for decoder
        cls_out = x[:, 0:1, :]                     # global token

        # learned queries
        q = self.token_queries.expand(B, -1, -1)   # [B,T-1,D]
        tokens = self.decoder(q, patch_tokens)     # [B,T-1,D]
        tokens = tokens + self.query_refiner(tokens)

        # concat global CLS with local tokens
        tokens = torch.cat([cls_out, tokens], dim=1)  # [B, T, D]

        return self.ln_out(self.proj_out(tokens))
    
    
class TwoStreamDenoiser(nn.Module):
    def __init__(
        self,
        num_points: int = 1024,
        num_latents: int = 256,
        cond_drop_prob: float = 0.1,
        input_channels: int = 3,
        output_channels: int = 3,
        latent_dim: int = 768,
        x_dim: int = 512,
        num_blocks: int = 6,
        num_compute_layers: int = 4,
        num_classes: int = 16,
        num_heads: int = 8,
        num_tokens_ppcd: int = 64,
        num_tokens_depth: int = 32,
        active_modalities: list = ["class", "view", "partial_pcd", "depth"],
    ):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.cond_drop_prob = cond_drop_prob
        self.active_modalities = active_modalities

        self.denoiser_backbone = Denoiser_backbone(
            input_channels=input_channels, output_channels=output_channels,
            num_x=num_points, num_z=num_latents, z_dim=latent_dim, x_dim=x_dim,
            num_blocks=num_blocks, num_compute_layers=num_compute_layers, num_heads=num_heads
        )

        # Encoders for active modalities
        self.encoders = nn.ModuleDict()
        self.token_type_ids = []

        modality_to_token_info = {
            "class": (0, 1, ClassEmbedding(num_classes=num_classes, embed_dim=latent_dim)) if "class" in active_modalities else None,
            "view": (1, 1, ViewAngleEmbedding(input_dim=3, embed_dim=latent_dim)) if "view" in active_modalities else None,
            "partial_pcd": (2, num_tokens_ppcd, PartialPointCloudEncoder(embed_dim=latent_dim, num_tokens=num_tokens_ppcd)) if "partial_pcd" in active_modalities else None,
            "depth": (3, num_tokens_depth, DepthMapEncoder(in_channels=1, embed_dim=latent_dim, num_tokens=num_tokens_depth)) if "depth" in active_modalities else None,
        }

        for modality in self.active_modalities:
            token_id, count, encoder = modality_to_token_info[modality]
            self.encoders[modality] = encoder
            self.token_type_ids.append((token_id, count))

        # Create token type embedding
        self.token_type_embeddings = nn.Embedding(4, latent_dim)  # Max 4 modalities (fixed IDs)
        nn.init.normal_(self.token_type_embeddings.weight, std=0.005)

        # Precompute token types template for active modalities
        token_type_list = []
        for token_id, count in self.token_type_ids:
            token_type_list += [token_id] * count
        self.register_buffer("token_types_template", torch.tensor(token_type_list, dtype=torch.long))

    def cached_model_kwargs(self, batch_size, model_kwargs):
        return model_kwargs

    def forward(self, x, t, class_labels=None, viewpoints=None, partial_pcd=None,
                 depth_maps=None, prev_latent=None):
        
        assert x.shape[-1] == self.num_points, f"Input point cloud must have {self.num_points} points, got {x.shape[-1]} points."

        B = x.shape[0]
        cond_tokens = []

        # Gather conditioning embeddings
        input_dict = {
            "class": class_labels,
            "view": viewpoints,
            "partial_pcd": partial_pcd,
            "depth": depth_maps
        }

        split_sizes = []
        for key in self.active_modalities:
            value = input_dict[key]
            if key == "class":
                tokens = self.encoders[key](value) if (value is not None and not torch.all(value == 0)) else torch.zeros(B, 1, self.latent_dim, device=x.device)
            elif key == "view":
                tokens = self.encoders[key](value) if (value is not None and not torch.all(value == 0)) else torch.zeros(B, 1, self.latent_dim, device=x.device)
            elif key == "partial_pcd":
                tokens = self.encoders[key](value) if (value is not None and not torch.all(value == 0)) else torch.zeros(B, self.encoders[key].num_tokens, self.latent_dim, device=x.device)
            elif key == "depth":
                tokens = self.encoders[key](value) if (value is not None and not torch.all(value == 0)) else torch.zeros(B, self.encoders[key].num_tokens, self.latent_dim, device=x.device) 
            cond_tokens.append(tokens)
            split_sizes.append(tokens.shape[1])

        cond_vec = torch.cat(cond_tokens, dim=1)
        token_types = self.token_types_template.unsqueeze(0).expand(B, -1).to(x.device)
        type_embeddings = self.token_type_embeddings(token_types)

        if self.training:
            cond_vec = cond_vec + type_embeddings

            # Full and Per-modality dropout for CFG
            full_drop_mask = torch.rand(B) < self.cond_drop_prob
            cond_keep_mask = torch.rand(B, len(split_sizes)) >= (self.cond_drop_prob)

            cond_keep_mask[full_drop_mask] = False
            cond_vec_chunks = torch.split(cond_vec, split_sizes, dim=1)

            masked_chunks = [
                chunk * cond_keep_mask[:, i].unsqueeze(1).unsqueeze(2).to(x.device)
                for i, chunk in enumerate(cond_vec_chunks)
            ]
            cond_vec = torch.cat(masked_chunks, dim=1)
        
        else:
            # Inference: Per-modality mask.
            cond_mask_chunks = []
            for key, size in zip(self.active_modalities, split_sizes):
                use_cond = input_dict[key] is not None and not torch.all(input_dict[key] == 0)
                mask_value = 1.0 if use_cond else 0.0
                cond_mask_chunks.append(torch.full((B, size, 1), mask_value, device=x.device))
            type_mask = torch.cat(cond_mask_chunks, dim=1)
            cond_vec = cond_vec + (type_embeddings * type_mask)
        

        # Pass to backbone
        x_denoised, latent = self.denoiser_backbone(
            x.permute(0, 2, 1).contiguous(), t, cond=cond_vec, prev_latent=prev_latent
        )
        x_denoised = x_denoised.permute(0, 2, 1).contiguous()
        return x_denoised, latent
