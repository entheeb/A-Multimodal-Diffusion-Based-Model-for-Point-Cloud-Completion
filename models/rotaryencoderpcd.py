import torch
import torch.nn as nn
import math


def apply_rotary_pos_emb(q, k, coords):
    """
    Applies 3D rotary positional embedding only to the first 6 dimensions (3 axis pairs)
    of the query and key projections.
    """
    theta = coords * math.pi               # [B, N, 3], scaled to [-π, π]
    sin, cos = theta.sin(), theta.cos()   # [B, N, 3]
    sin = sin.unsqueeze(1)                # [B, 1, N, 3]
    cos = cos.unsqueeze(1)

    def rotate(x):
        x_rot = x[..., :6]                # First 6 dims only
        x_rest = x[..., 6:]               # Remaining dims stay unchanged
        x1 = x_rot[..., 0::2]             # Even indices: x, y, z
        x2 = x_rot[..., 1::2]             # Odd indices
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return torch.cat([x_rotated, x_rest], dim=-1)

    return rotate(q), rotate(k)

'''def apply_rotary_pos_emb(q, k, coords):
    """
    Apply RoPE to first 12 dims (6 pairs) using 3D coords
    Assumes D_head >= 12 and divisible by 2
    """
    theta = coords * math.pi
    sin, cos = theta.sin(), theta.cos()
    sin = sin.unsqueeze(1)  # [B, 1, N, 3]
    cos = cos.unsqueeze(1)

    def rotate(x):
        x_rot = x[..., :12]               # First 12 dims = 6 pairs
        x_rest = x[..., 12:]
        x1 = x_rot[..., 0::2]             # even dims → [B, H, N, 6]
        x2 = x_rot[..., 1::2]             # odd dims → [B, H, N, 6]

        # Repeat (x, y, z) twice to match 6 pairs
        sin_expanded = sin.repeat(1, x1.shape[1], 1, 2)  # → [B, H, N, 6]
        cos_expanded = cos.repeat(1, x1.shape[1], 1, 2)

        x_rotated = torch.cat([
            x1 * cos_expanded - x2 * sin_expanded,
            x1 * sin_expanded + x2 * cos_expanded
        ], dim=-1)
        return torch.cat([x_rotated, x_rest], dim=-1)

    return rotate(q), rotate(k)'''


class RotarySelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos):
        B, N, D = x.shape
        H = self.heads
        D_head = D // H
        assert D_head >= 6 and D_head % 2 == 0, "Head dimension must be even and ≥ 6 for RoPE"

        qkv = self.qkv(x).reshape(B, N, 3, H, D_head).permute(2, 0, 3, 1, 4)  # [3, B, H, N, D_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = apply_rotary_pos_emb(q, k, pos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class RotaryTransformerLayer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotarySelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


class PartialPointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=256, num_tokens=256, num_layers=6, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.GELU()
        )

        self.encoder = nn.ModuleList([
            RotaryTransformerLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        self.decoder_attn = RotarySelfAttention(embed_dim, num_heads, dropout=dropout)

        self.refiner = nn.ModuleList([
            RotaryTransformerLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers // 2)
        ])

        self.token_proj = nn.Linear(embed_dim, embed_dim)
        self.ln_output = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
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
        # pcd: [B, N, 3]
        B, N, _ = pcd.shape
        x = self.input_proj(pcd)  # [B, N, D]

        for layer in self.encoder:
            x = layer(x, pos=pcd)

        queries = self.token_queries.expand(B, -1, -1)  # [B, T, D]
        tokens = self.decoder_attn(queries, pcd)

        for layer in self.refiner:
            tokens = layer(tokens, pos=pcd)

        return self.ln_output(self.token_proj(tokens))  # [B, T, D] 
    



'''class PartialPointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=256, num_tokens=256, num_layers=6, num_heads=8, num_freqs=8):
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
            nn.Linear(in_dim_eff, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.GELU(),
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

        # Query refinement
        query_refiner_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.query_refiner = nn.TransformerEncoder(query_refiner_layer, num_layers=num_layers // 2)

        
        # Final linear projection and LayerNorm
        self.ln_output = nn.LayerNorm(embed_dim)
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
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
        # pcd: [B, N, 3]
        B, N, _ = pcd.shape

        pe = self.pos_enc(pcd)            # [B, N, 6F]
        x = self.input_proj(torch.cat([pcd, pe], dim=-1)) # concat xyz+PE  -> [B,N,3+6F]
        #x = self.input_proj(pcd)          # [B, N, D]
        #x = self.dropout(x)           # [B, N, D]

        # Transformer encoder: mix point features
        encoded = self.encoder(x)  # [B, N, D]

        # Use token queries to attend to point tokens via dot-product attention
        # token_queries: [1, T, D] → broadcast to [B, T, D]
        queries = self.token_queries.expand(B, -1, -1)  # [B, T, D]
        tokens = self.decoder(queries, encoded)         # [B, T, D]
        refined = self.query_refiner(tokens + queries)
        tokens = tokens + refined            # [B, T, D] (refine queries)

        return self.token_proj(self.ln_output(tokens))  # [B, T, D]'''
