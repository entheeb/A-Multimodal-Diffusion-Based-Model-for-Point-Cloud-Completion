# run.py

import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import open3d as o3d

from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.model import TwoStreamDenoiser
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from point_e.dataset.multimodal_dataloader import MultiModalDataset
from point_e.util.point_cloud import PointCloud
from point_e.dataset.mvp_dataloader import MVP_CP 
from point_e.dataset.modelnet_dataloader import ModelnetDataset


def set_seed(seed):
    seed = seed - 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_pcd(points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_target_point_clouds(batch_target_points, out_dir, colors=None):
    os.makedirs(out_dir, exist_ok=True)
    for i, pts in enumerate(batch_target_points):
        pcd = to_pcd(points= pts.cpu().numpy(), colors=colors)
        o3d.io.write_point_cloud(os.path.join(out_dir, f"target_{i + 1}_shapenet.ply"), pcd)


def load_model(cfg, device):
    model = TwoStreamDenoiser(**cfg.model).to(device)
    checkpoint_path = cfg.sample.load_checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def save_samples(samples, output_dir, save_format):
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        sample_np = sample.cpu().T.numpy()  # [N, C], where C is 3 (XYZ) or 6 (XYZ+RGB)

        # Split XYZ
        coords = sample_np[:, :3].clip(-1, 1)  # Clip XYZ in [-1, 1] range

        # Handle RGB if exists
        channels = {}
        if sample_np.shape[1] >= 6:
            rgb = sample_np[:, 3:6]
            # Clip RGB in [0, 1] scale
            rgb = np.clip(rgb, -1, 1)
            rgb = (rgb + 1) / 2     # Scale to [0, 1]
            '''channels = {
                "R": rgb[:, 0],
                "G": rgb[:, 1],
                "B": rgb[:, 2],
            }'''
        else:
            rgb = None
        
        target_pcd = to_pcd(points=coords, colors=rgb)
        o3d.io.write_point_cloud(f"/home/obaidah/point-e/point_e/samples_modelnet/pred_{i + 1}_shapenet.ply", target_pcd)

        '''# Create PointCloud object
        pc = PointCloud(coords=coords, channels=channels)

        # Save based on format
        path = os.path.join(output_dir, f"sample_{i:03d}_colored.{save_format}")
        if save_format == "npz":
            pc.save(path)
        elif save_format == "ply":
            with open(path, "wb") as f:
                pc.write_ply(f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")'''

def build_dataloader(cfg):
    dataset = MultiModalDataset(h5_path=cfg.data.h5_path)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.sample.num_samples,
        sampler=None,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=False,
    )
    return dataloader

def build_dataloader_mvp(cfg):
    dataset = MVP_CP(prefix="train", n_samples=1024)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.sample.num_samples,
        sampler=None,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=False,
    )
    return dataloader


def build_modelnet_dataloader(cfg):
    dataset = ModelnetDataset(h5_path=cfg.data.h5_path,)


    dataloader = DataLoader(
        dataset,
        batch_size=cfg.sample.num_samples,
        sampler=None,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )

    return dataloader





def main(cfg):
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and diffusion
    model = load_model(cfg, device)
    steps = cfg.diffusion.timesteps
    schedule = cfg.diffusion.schedule
    betas = get_named_beta_schedule(schedule, steps)
    diffusion = GaussianDiffusion(**cfg.diffusion.gaussiandiffusion, betas=betas)
    dataloader= build_modelnet_dataloader(cfg)

    # Setup sampler
    sampler = PointCloudSampler(
        device=device,
        models=[model],
        diffusions=[diffusion],
        num_points=[4096],
        aux_channels=[],
        guidance_scale=[cfg.sample.guidance_scale],
        clip_denoised= True,
        use_karras=[cfg.sample.use_karras],
        karras_steps=[cfg.sample.karras_steps],
        sigma_min=[cfg.sample.sigma_min],
        sigma_max=[cfg.sample.sigma_max],
        s_churn=[cfg.sample.s_churn],
    )

    class_labels, partial_pcd, depth_maps, viewpoints, target_points = next(iter(dataloader))
    #class_labels, partial_pcd, target_points = next(iter(dataloader))
    #x_target = target_points.clone().permute(0, 2, 1).contiguous()
    #x_target = x_target.to(device).float()  # [B, C, N]
    
    save_target_point_clouds(target_points, "/home/obaidah/point-e/point_e/target_samples_modelnet")
    save_target_point_clouds(partial_pcd, "/home/obaidah/point-e/point_e/partial_samples_modelnet")

    model_kwargs = {
                    "class_labels": class_labels.to(device),                 # [B]
                    "viewpoints": viewpoints.to(device),             # [B, 3]
                    "partial_pcd": partial_pcd.to(device),            # [B, 1024, 3]
                    "depth_maps": depth_maps.to(device).unsqueeze(1),  # [B, 1, 512, 512]
                }

    # Sampling
    print("Sampling point clouds...")
    all_samples = []
    for sample in tqdm(sampler.sample_batch_progressive(cfg.sample.num_samples, model_kwargs, x_target=None)):
        all_samples.append(sample)

    final_samples = all_samples[-1]
    print(f"Sampled {len(final_samples)} point clouds.")
    print(f"Final samples shape: {final_samples.shape}")
    #for i, name in enumerate(['R', 'G', 'B']):
       #print(f"{name} max: {final_samples[:, 3+i, :].max().item():.4f}, min: {final_samples[:, 3+i, :].min().item():.4f}")


    # Save to disk
    print("Saving samples...")
    save_samples(final_samples, cfg.sample.output_dir, cfg.sample.save_format)
    print(f"Saved to {cfg.sample.output_dir}")


if __name__ == "__main__":
    cfg = OmegaConf.load("/home/obaidah/point-e/point_e/config.yaml")
    main(cfg)
    print("Finished sampling.")
