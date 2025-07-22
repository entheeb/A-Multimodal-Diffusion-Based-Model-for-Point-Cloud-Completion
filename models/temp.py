obj_type, obj_id, scan_idx = self.idx_list[idx]
        with h5py.File(self.h5_path, "r") as f:
            obj_data = f[obj_type][obj_id]
            partial_pcd = torch.from_numpy(obj_data["points"][scan_idx][:])
            depth_maps = torch.from_numpy(obj_data["depth_maps"][scan_idx][:]).float()
            viewpoints = torch.from_numpy(obj_data["viewpoints"][scan_idx][:]).float()
            target_points = torch.from_numpy(obj_data["target_points"][:])
            #label_idx = torch.LongTensor([label_dict[obj_data.attrs.get("label", None)]])
            label_idx = torch.tensor(label_dict[obj_data.attrs.get("label", None)], dtype=torch.long)


class PartialPointCloudEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, embed_dim: int = 768, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, pcd):
        B, N, _ = pcd.shape
        x = self.encoder(pcd)
        x = x.view(B, self.num_tokens, -1)
        return x
    

'''depth_min=78, depth_max=255, viewpoints_max_abs=2.8776512145996094'''

    '''def load_dataset(self):
        with h5py.File(self.h5_path, "r") as f:
            self.obj_types = list(f.keys())
            self.idx_list = []
            for obj_type in self.obj_types:
                obj_indices = [[obj_type, obj_id] for obj_id in f[obj_type].keys() if (obj_type, obj_id) not in self.skip_list else continue]
                for obj_idx in obj_indices:
                    self.idx_list += [[obj_idx[0], obj_idx[1], i] for i in range(self.num_scans)]'''
    
    def load_dataset(self):
        with h5py.File(self.h5_path, "r") as f:
            self.obj_types = list(f.keys())
            self.idx_list = []
            for obj_type in self.obj_types:
                for obj_id in f[obj_type].keys():
                    try:
                        if len(f[obj_type][obj_id]["depth_maps"]) == self.num_scans:
                            self.idx_list += [[obj_type, obj_id, i] for i in range(self.num_scans)]
                        else:
                            print(f"[SKIP] Skipping {obj_type}/{obj_id} (has {len(f[obj_type][obj_id]['depth_maps'])} scans)")
                    except KeyError:
                        print(f"[SKIP] {obj_type}/{obj_id} missing depth_maps key")



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
    


# --- Condition Embedding Modules ---

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, class_labels):
        return self.embedding(class_labels).unsqueeze(1)


class ViewAngleEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU()
        )

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
        return self.mlp(view_angles).unsqueeze(1)


class PartialPointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=768, num_tokens=4, num_layers=4, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Project 3D points to embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Token selection: learned queries that extract [num_tokens] tokens from the encoded sequence
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # Final linear projection (optional, for refinement)
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.token_queries, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, pcd):
        # pcd: [B, N, 3]
        B, N, _ = pcd.shape

        x = self.input_proj(pcd)  # [B, N, D]

        # Transformer encoder: mix point features
        encoded = self.transformer(x)  # [B, N, D]

        # Use token queries to attend to point tokens via dot-product attention
        # token_queries: [1, T, D] → broadcast to [B, T, D]
        queries = self.token_queries.expand(B, -1, -1)  # [B, T, D]
        attn_scores = torch.matmul(queries, encoded.transpose(1, 2))  # [B, T, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T, N]
        tokens = torch.matmul(attn_weights, encoded)  # [B, T, D]

        return self.token_proj(tokens)  # [B, T, D]


class DepthMapEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 768, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((self.num_tokens, 1))
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.conv(depth_maps)  # [B, embed_dim, T, 1]
        x = x.squeeze(-1).transpose(1, 2)  # [B, T, embed_dim]
        return x
    


class TwoStreamDenoiser(nn.Module):
    '''
    Full Point diffusion model using MCC's encoders with the Two Stream backbone
    '''
    def __init__(
        self,
        num_points: int = 1024,
        num_latents: int = 256,
        cond_drop_prob: float = 0.1,
        input_channels: int = 6,
        output_channels: int = 6,
        latent_dim: int = 768,
        num_blocks: int = 6,
        num_compute_layers: int = 4,
        num_classes: int = 1,
        num_heads: int = 8,
        num_tokens_ppcd: int = 256,
        num_tokens_depth: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.cond_drop_prob = cond_drop_prob
        # define backbone
        self.denoiser_backbone = Denoiser_backbone(input_channels=input_channels, output_channels=output_channels, 
                                      num_x=num_points, num_z=num_latents, z_dim=latent_dim, 
                                      num_blocks=num_blocks, num_compute_layers=num_compute_layers, num_heads=num_heads)
     
        # Modality-specific encoders
        self.class_embed = ClassEmbedding(num_classes=num_classes, embed_dim=latent_dim)
        self.view_embed = ViewAngleEmbedding(input_dim=3, embed_dim=latent_dim)
        self.partial_pcd_encoder = PartialPointCloudEncoder(embed_dim=latent_dim, num_tokens=num_tokens_ppcd)
        self.depth_encoder = DepthMapEncoder(in_channels=1, embed_dim=latent_dim, num_tokens=num_tokens_depth)
        self.token_type_embeddings = nn.Embedding(4, latent_dim)  # 4 modalities: class, view, partial, depth

        # Precompute token types
        token_type_list = (
            [0] * 1 +                     # ClassEmbedding: 1 token
            [1] * 1 +                     # ViewEmbedding: 1 token
            [2] * self.partial_pcd_encoder.num_tokens +  # Partial PCD
            [3] * self.depth_encoder.num_tokens          # Depth Map
        )
        token_types_tensor = torch.tensor(token_type_list, dtype=torch.long)
        self.register_buffer("token_types_template", token_types_tensor)
        self.total_tokens = len(token_type_list)

    '''def cached_model_kwargs(self, model_kwargs):
        with torch.no_grad():
            cond_dict = {}
            images = preprocess_img(model_kwargs["images"])
            embeddings = self.mcc_encoder(
                images,
                model_kwargs["seen_xyz"],
                model_kwargs["seen_xyz_mask"],
            )
            cond_dict["embeddings"] = embeddings
            if "prev_latent" in model_kwargs:
                cond_dict["prev_latent"] = model_kwargs["prev_latent"]
            return cond_dict'''
    
    def cached_model_kwargs(self, batch_size, model_kwargs):
        return model_kwargs   

    def forward(
        self,
        x,                    # [B, C, N]
        t,                    # [B]
        class_labels=None,    # [B]
        viewpoints=None,     # [B, 3]
        partial_pcd=None,     # [B, 1024, 3]
        depth_maps=None,      # [B, 1, 512, 512]
        prev_latent=None      # [B, num_z + num_cond + 1, z_dim]
    ):
        """
        Forward pass through the model.

        Parameters:
        x: Tensor of shape [B, C, N_points], raw input point cloud.
        t: Tensor of shape [B], time step.
        images (Tensor, optional): A batch of images to condition on.
        seen_xyz (Tensor, optional): A batch of xyz maps to condition on.
        seen_xyz_mask (Tensor, optional): Validity mask for xyz maps.
        embeddings (Tensor, optional): A batch of conditional latent (avoid duplicate 
                                        computation of MCC encoder in diffusion inference)
        prev_latent (Tensor, optional): Self-conditioning latent.

        Returns:
        x_denoised: Tensor of shape [B, C, N_points], denoised point cloud/noise.
        """
        assert x.shape[-1] == self.num_points

        B = x.shape[0]
        cond_tokens = []

        # Prepare class label embedding
        if class_labels is not None and not torch.all(class_labels == 0):
            class_embed = self.class_embed(class_labels)
        else:
            class_embed = torch.zeros(B, 1, self.latent_dim, device=x.device)
        cond_tokens.append(class_embed)

        # View angle embedding
        if viewpoints is not None and not torch.all(viewpoints == 0):
            view_embed = self.view_embed(viewpoints)
        else:
            view_embed = torch.zeros(B, 1, self.latent_dim, device=x.device)
        cond_tokens.append(view_embed)

        # Partial point cloud
        if partial_pcd is not None and not torch.all(partial_pcd == 0):
            partial_embed = self.partial_pcd_encoder(partial_pcd)
        else:
            partial_embed = torch.zeros(B, self.partial_pcd_encoder.num_tokens, self.latent_dim, device=x.device)
        cond_tokens.append(partial_embed)

        # Depth map embedding
        if depth_maps is not None and not torch.all(depth_maps == 0):
            depth_embed = self.depth_encoder(depth_maps)
        else:
            depth_embed = torch.zeros(B, self.depth_encoder.num_tokens, self.latent_dim, device=x.device)
        cond_tokens.append(depth_embed)

        # Stack conditions: [B, total_tokens, latent_dim]
        cond_vec = torch.cat(cond_tokens, dim=1)

        token_types = self.token_types_template.unsqueeze(0).expand(B, -1).to(x.device)  # [B, total_tokens]


        # Lookup token type embeddings
        type_embeddings = self.token_type_embeddings(token_types)  # [B, total_tokens, latent_dim]

        if self.training:
            cond_vec = cond_vec + type_embeddings

        else:
            # A per-condition active mask (True = condition is given, False = zeros)
            condition_masks = [
                class_labels is not None and not torch.all(class_labels == 0),
                viewpoints is not None and not torch.all(viewpoints == 0),
                partial_pcd is not None and not torch.all(partial_pcd == 0),
                depth_maps is not None and not torch.all(depth_maps == 0),
            ]

            # Generate a token-wise binary mask: [B, total_tokens, 1]
            split_sizes = [
                class_embed.shape[1],        # Class token count
                view_embed.shape[1],         # Viewpoint token count
                partial_embed.shape[1],      # Partial point cloud token count
                depth_embed.shape[1],        # Depth map token count
            ]

            cond_mask_chunks = []
            for i, size in enumerate(split_sizes):
                mask_value = 1.0 if condition_masks[i] else 0.0
                cond_mask_chunks.append(torch.full((B, size, 1), mask_value, device=x.device))

            type_mask = torch.cat(cond_mask_chunks, dim=1)  # [B, total_tokens, 1]

            # Apply mask: add type embeddings only to active conditions
            cond_vec = cond_vec + (type_embeddings * type_mask)
        

        # condition dropout
        if self.training:

            # 1. Full condition dropout for some samples (classifier-free guidance)
            full_drop_mask = torch.rand(B) < self.cond_drop_prob  # True = drop all

            # 2. Per-condition dropout mask: [B, 4]
            # condition types: [class, view, partial_pcd, depth]
            cond_keep_mask = torch.rand(B, 4) >= (self.cond_drop_prob + 0.1)  # True = keep

            # 3. If full dropout is True for a sample, override all keeps to False
            cond_keep_mask[full_drop_mask] = False

            # 4. Split cond_vec and apply mask
            split_sizes = [
                class_embed.shape[1],        # Class token count 
                view_embed.shape[1],         # Viewpoint token count 
                partial_embed.shape[1],      # Partial point cloud token count 
                depth_embed.shape[1],        # Depth map token count 
            ] 
            cond_vec_chunks = torch.split(cond_vec, split_sizes, dim=1)

            masked_chunks = []
            for i, chunk in enumerate(cond_vec_chunks):
                # Create a mask for each condition: [B, 1, 1]
                mask_i = cond_keep_mask[:, i].unsqueeze(1).unsqueeze(2).to(chunk.device)
                masked_chunks.append(chunk * mask_i)

            cond_vec = torch.cat(masked_chunks, dim=1)

        # denoiser forward
        x_denoised, latent = self.denoiser_backbone(
            x.permute(0, 2, 1).contiguous(),  # [B, N, C]
            t,
            cond=cond_vec,
            prev_latent=prev_latent,
        )
        x_denoised = x_denoised.permute(0, 2, 1).contiguous() # [B, C, N]
        return x_denoised, latent
    


class PartialPointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=768, num_tokens=64, num_layers=6, num_heads=8, num_freqs=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # position encoding
        self.pos_enc = FourierPE(num_freqs=num_freqs)

        # Project 3D points to embed_dim
        pe_dim        = 6 * num_freqs      # 6F from FourierPE
        in_dim_eff    = input_dim + pe_dim
        self.input_proj = nn.Linear(in_dim_eff, embed_dim)
        self.pre_ln     = nn.LayerNorm(embed_dim)
        #self.dropout = nn.Dropout(0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Token selection: learned queries that extract [num_tokens] tokens from the encoded sequence
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # Final linear projection (optional, for refinement)
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.token_queries, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, pcd):
        # pcd: [B, N, 3]
        B, N, _ = pcd.shape

        pe = self.pos_enc(pcd)            # [B, N, 6F]
        x = self.input_proj(torch.cat([pcd, pe], dim=-1)) # concat xyz+PE  -> [B,N,3+6F]
        x = self.pre_ln(x)
        #x = self.dropout(x)           # [B, N, D]

        # Transformer encoder: mix point features
        encoded = self.transformer(x)  # [B, N, D]

        # Use token queries to attend to point tokens via dot-product attention
        # token_queries: [1, T, D] → broadcast to [B, T, D]
        queries = self.token_queries.expand(B, -1, -1)  # [B, T, D]
        attn_scores = torch.matmul(queries, encoded.transpose(1, 2))  # [B, T, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T, N]
        tokens = torch.matmul(attn_weights, encoded)  # [B, T, D]

        return self.token_proj(tokens)  # [B, T, D]
    



    '''class PartialPointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=768, num_tokens=64, num_layers=6, num_heads=8, num_freqs=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # position encoding
        self.pos_enc = FourierPE(num_freqs=num_freqs)

        # Project 3D points to embed_dim
        pe_dim        = 6 * num_freqs      # 6F from FourierPE
        in_dim_eff    = input_dim + pe_dim

        # Pre-MLP + LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim_eff, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        #self.dropout = nn.Dropout(0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder (query refinement)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=0.0
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)

        # Token selection: learned queries that extract [num_tokens] tokens from the encoded sequence
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, embed_dim))
        # Token positions: learnable positional embeddings for each token
        self.token_positions = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # Final linear projection (optional, for refinement)
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.token_queries, std=0.02)
        nn.init.normal_(self.token_positions, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, pcd):
        # pcd: [B, N, 3]
        B, N, _ = pcd.shape

        pe = self.pos_enc(pcd)            # [B, N, 6F]
        x = self.input_proj(torch.cat([pcd, pe], dim=-1)) # concat xyz+PE  -> [B,N,3+6F]
        #x = self.dropout(x)           # [B, N, D]

        # Transformer encoder: mix point features
        encoded = self.encoder(x)  # [B, N, D]

        # Use token queries to attend to point tokens via dot-product attention
        # token_queries: [1, T, D] → broadcast to [B, T, D]
        queries = self.token_queries + self.token_positions
        queries = queries.expand(B, -1, -1)  # [B, T, D]
        tokens = self.decoder(queries, encoded)         # [B, T, D]

        return self.token_proj(tokens)  # [B, T, D]'''
    







    '''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierPE(nn.Module):
    def __init__(self, num_freqs: int = 8, scale: float = 1.0):
        super().__init__()
        freqs = 2. ** torch.arange(num_freqs) * math.pi / scale
        self.register_buffer("freqs", freqs)

    def forward(self, xyz):
        enc = xyz.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)
        return enc.flatten(-2)


def knn_aggregate(xyz, feats, k=16):
    B, N, D = feats.shape
    with torch.no_grad():
        dists = torch.cdist(xyz, xyz)
        knn_idx = dists.topk(k=k+1, dim=-1, largest=False).indices[..., 1:]

    knn_feats = torch.gather(
        feats.unsqueeze(2).expand(-1, -1, N, -1), 2,
        knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    )
    return knn_feats.mean(dim=2)


class LocalPointNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, grouped_feats):
        B, G, K, D = grouped_feats.shape
        grouped_feats = grouped_feats.reshape(B * G, K, D)
        out = self.mlp(grouped_feats)
        out = out.max(dim=1).values
        return out.reshape(B, G, -1)


def fps(xyz, npoints):
    B, N, _ = xyz.shape
    indices = torch.rand(B, npoints, device=xyz.device).mul(N).long()
    return indices


def group_knn(xyz, centers, k):
    B, N, _ = xyz.shape
    B, G, _ = centers.shape
    dists = torch.cdist(centers, xyz)
    knn_idx = dists.topk(k=k, dim=-1, largest=False).indices
    return knn_idx


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Linear(dim, 1)

    def forward(self, x):
        attn_scores = self.attention(x)  # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N, 1]
        pooled = (x * attn_weights).sum(dim=1)  # [B, D]
        return pooled


class PartialPointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=256, num_tokens=256, num_layers=6, num_heads=8, num_freqs=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.pos_enc = FourierPE(num_freqs=num_freqs)
        pe_dim = 6 * num_freqs
        in_dim_eff = input_dim + pe_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim_eff, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = AttentionPooling(embed_dim)

        self.query_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_tokens * embed_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=0.0
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)

        query_refiner_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0
        )
        self.query_refiner = nn.TransformerEncoder(query_refiner_layer, num_layers=num_layers // 2)

        self.ln_output = nn.LayerNorm(embed_dim)
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
    nn.init.constant_(self.token_proj.weight, 0)
    nn.init.constant_(self.token_proj.bias, 0)
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
        pe = self.pos_enc(pcd)
        x = self.input_proj(torch.cat([pcd, pe], dim=-1))

        encoded = self.encoder(x)

        global_feat = self.attention_pool(encoded)
        queries = self.query_generator(global_feat).view(B, self.num_tokens, self.embed_dim)
        tokens = self.decoder(queries, encoded)
        refined = self.query_refiner(tokens + queries)
        tokens = tokens + refined
        return self.ln_output(self.token_proj(tokens))
'''



"""import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DepthMapEncoder(nn.Module):

    def __init__(self, in_channels: int = 1, embed_dim: int = 256,
                 num_tokens: int = 64, patch: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.patch = patch

        # patch projection
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)

        # fixed 2‑D sin‑cos positional encoding
        h = w = 512 // patch
        pe_mat = build_2d_sincos_position_embedding(h, w, embed_dim)
        self.register_buffer("pos_embed", pe_mat)

        # Transformer mixer over patch tokens (+ CLS)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8,
                                               batch_first=True, dropout=0.0)
        self.mixer = nn.TransformerEncoder(enc_layer, num_layers=4)

        # global CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        #learned query tokens
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # decoder to extract latent geometry tokens
        dec_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8,
                                               batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

        query_refiner_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8,
                                                         batch_first=True,
                                                         dropout=0.0)
        self.query_refiner = nn.TransformerEncoder(query_refiner_layer,
                                                   num_layers=2)

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
                if m is not self.proj_out:  # already done
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

        x = self.norm(x)
        x = self.mixer(x)                          # CLS updated

        patch_tokens = x[:, 1:, :]                 # discard CLS for decoder
        cls_out = x[:, 0:1, :]                     # global token

        # learned queries
        q = self.token_queries.expand(B, -1, -1)   # [B,T,D]
        tokens = self.decoder(q, patch_tokens)     # [B,T,D]
        tokens = tokens + self.query_refiner(tokens + q)

        # concat global CLS with local tokens
        tokens = torch.cat([cls_out, tokens], dim=1)  # [B, 1+T, D]

        return self.ln_out(self.proj_out(tokens))
"""