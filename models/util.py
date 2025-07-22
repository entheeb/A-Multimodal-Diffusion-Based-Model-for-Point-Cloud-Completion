import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import open3d as o3d
import os


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MCC: https://github.com/facebookresearch/MCC
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# --------------------------------------------------------
# Timestep embedding
# References:
# Point-E: https://github.com/openai/point-e
# --------------------------------------------------------
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# --------------------------------------------------------
# Image preprocessor
# References:
# MCC: https://github.com/facebookresearch/MCC
# --------------------------------------------------------
def preprocess_img(x):
    """
    Preprocess images for MCC encoder.
    """
    if x.shape[2] != 224:
        x = F.interpolate(
            x,
            scale_factor=224./x.shape[2],
            mode="bilinear",
        )
    resnet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape((1, 3, 1, 1))
    resnet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape((1, 3, 1, 1))
    imgs_normed = (x - resnet_mean) / resnet_std
    return imgs_normed


def build_warm_cosine_scheduler(
    optimizer,
    total_steps: int,
    peak_lr: float = 3e-4,
    warmup_ratio: float = 0.05,
    lr_floor: float = 0.01,
    last_epoch: int = -1,
):
    """
    Linear warm-up ➜ cosine decay that bottoms out at `lr_floor × peak_lr`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser whose LR will be scheduled.
    total_steps : int
        Total optimiser steps for the run.
    peak_lr : float
        Peak learning rate (must match optimizer's initial LR).
    warmup_ratio : float, optional
        Fraction of total steps used for linear warm-up. Default 0.05 (5%).
    lr_floor : float, optional
        Final LR as a fraction of peak LR. Default 0.01 (1%).
    last_epoch : int, optional
        The index of the last completed step. Set to global_step - 1 when resuming.

    Returns
    -------
    torch.optim.lr_scheduler.SequentialLR
    """
    assert total_steps > 0, "Total steps must be greater than 0."
    warmup_steps = max(1, int(warmup_ratio * total_steps))  # prevent 0 warmup

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=peak_lr * lr_floor,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
        last_epoch=last_epoch
    )
    return scheduler 


def to_pcd(points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_samples(samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        sample_np = sample.cpu().T.numpy()
        coords = sample_np[:, :3].clip(-1, 1)
        rgb = sample_np[:, 3:6] if sample_np.shape[1] >= 6 else None
        if rgb is not None:
            rgb = np.clip((rgb + 1) / 2, 0, 1)
        pcd = to_pcd(points=coords, colors=rgb)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"sample_{i + 1}.ply"), pcd)

def save_target_point_clouds(batch_target_points, out_dir, colors=None, prefix="target"):
    os.makedirs(out_dir, exist_ok=True)
    for i, pts in enumerate(batch_target_points):
        pcd = to_pcd(points= pts.cpu().numpy(), colors=colors)
        o3d.io.write_point_cloud(os.path.join(out_dir, f"{prefix}_{i + 1}.ply"), pcd)
