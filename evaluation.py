import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import logging
from collections import defaultdict
from pointnet.utils import farthest_point_sampling

from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.model import TwoStreamDenoiser
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from point_e.dataset.modelnet_dataloader import ModelnetDatasetTest
from point_e.models.util import chamfer_distance_xyz, fscore_point_cloud_batch


def setup_logger(log_file):
    logger = logging.getLogger("evaluation_logger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1).repeat(1, idx.shape[1])
    return points[batch_indices, idx, :]


def batch_fps_tensor(pcd_tensor, n_samples=1024):
    fps_idx = farthest_point_sampling(pcd_tensor, n_samples)
    return index_points(pcd_tensor, fps_idx)


def load_model(cfg, device, logger):
    model = TwoStreamDenoiser(**cfg.model).to(device)
    checkpoint_path = cfg.sample.load_checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logger.info(f"Loading model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def build_modelnet_dataloader(cfg):
    dataset = ModelnetDatasetTest(h5_path=cfg.data.h5_path)
    idx_to_class = {v: k for k, v in dataset.class_to_new_label.items()}
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.sample.num_samples,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, idx_to_class


def main(cfg):
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = os.path.join(os.getcwd(), "evaluation_log_8192.txt")
    logger = setup_logger(log_file)

    logger.info(f"Using device: {device}")
    logger.info("Starting evaluation...")

    # Log full config once
    logger.info("=== Loaded Configuration ===")
    logger.info(OmegaConf.to_yaml(cfg))  

    model = load_model(cfg, device, logger)
    betas = get_named_beta_schedule(cfg.diffusion.schedule, cfg.diffusion.timesteps)
    diffusion = GaussianDiffusion(**cfg.diffusion.gaussiandiffusion, betas=betas)
    dataloader, idx_to_class = build_modelnet_dataloader(cfg)

    sampler = PointCloudSampler(
        device=device,
        models=[model],
        diffusions=[diffusion],
        num_points=[8192],
        aux_channels=[],
        guidance_scale=[cfg.sample.guidance_scale],
        clip_denoised=True,
        use_karras=[cfg.sample.use_karras],
        karras_steps=[cfg.sample.karras_steps],
        sigma_min=[cfg.sample.sigma_min],
        sigma_max=[cfg.sample.sigma_max],
        s_churn=[cfg.sample.s_churn],
    )

    total_metrics = defaultdict(list)
    class_metrics = defaultdict(lambda: defaultdict(list))

    model_kwargs_logged = False  # Flag to log model_kwargs only once

    for batch_idx, (class_labels, partial_pcd, depth_maps, viewpoints, target_points) in enumerate(tqdm(dataloader)):
        target_points = target_points.to(device)

        model_kwargs = {
            "class_labels": class_labels.to(device),
            "viewpoints": viewpoints.to(device),
            "partial_pcd": partial_pcd.to(device),
            "depth_maps": depth_maps.to(device).unsqueeze(1),
        }

         # Log model_kwargs once
        if not model_kwargs_logged:
            logger.info("=== Sample model_kwargs ===")
            for k, v in model_kwargs.items():
                logger.info(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            model_kwargs_logged = True

        all_samples = []
        for sample in sampler.sample_batch_progressive(cfg.sample.num_samples, model_kwargs, x_target=None):
            all_samples.append(sample)
        pred = all_samples[-1].to(device).clamp(-0.5, 0.5)

        B, C, N = pred.shape
        logger.info(f"Predicted point cloud resolution: {N} points per sample")

        pred_xyz = pred[:, :3, :].transpose(1, 2).contiguous()
        gt_xyz = target_points

        # Full-resolution metrics
        cd_full = chamfer_distance_xyz(pred, gt_xyz.permute(0, 2, 1))
        #cd_full_factor = chamfer_distance_xyz_with_factor(pred, gt_xyz.permute(0, 2, 1))
        f1_full, _, _ = fscore_point_cloud_batch(pred_xyz, gt_xyz)
        #f1_squared_full, _, _ = fscore_point_cloud_batch_squared(pred_xyz, gt_xyz)

        for name, value in zip(
            ["cd_full",  "f1_full"],
            [cd_full,  f1_full]
        ):
            total_metrics[name].extend(value.cpu().tolist())
            for i, cls in enumerate(class_labels):
                class_metrics[int(cls)][name].append(value[i].item())

        if N > 1024:
            pred_fps = batch_fps_tensor(pred_xyz, 1024)
            gt_fps = gt_xyz  #batch_fps_tensor(gt_xyz, 1024)

            cd_fps = chamfer_distance_xyz(pred_fps.permute(0, 2, 1), gt_fps.permute(0, 2, 1))
            #cd_fps_factor = chamfer_distance_xyz_with_factor(pred_fps.permute(0, 2, 1), gt_fps.permute(0, 2, 1))
            f1_fps, _, _ = fscore_point_cloud_batch(pred_fps, gt_fps)
            #f1_squared_fps, _, _ = fscore_point_cloud_batch_squared(pred_fps, gt_fps)

            for name, value in zip(
                ["cd_fps",  "f1_fps"],
                [cd_fps, f1_fps]
            ):
                total_metrics[name].extend(value.cpu().tolist())
                for i, cls in enumerate(class_labels):
                    class_metrics[int(cls)][name].append(value[i].item())

            logger.info(
                f"Batch {batch_idx} | CD: {cd_full.mean():.6f} / {cd_fps.mean():.6f} | "
                f"F1: {f1_full.mean():.6f} / {f1_fps.mean():.6f} | "
            )
        else:
            logger.info(
                f"Batch {batch_idx} | CD: {cd_full.mean():.6f} | "
                f"F1: {f1_full.mean():.6f} | — FPS skipped"
            )

    logger.info("=== Overall Metrics ===")
    for name, values in total_metrics.items():
        logger.info(f"{name}: {np.mean(values):.6f}")

    logger.info("=== Per-Class Metrics ===")
    for cls, metrics in class_metrics.items():
        class_name = idx_to_class.get(cls, f"Unknown_{cls}")
        logs = [f"Class {cls} ({class_name}):"]
        for metric_name, values in metrics.items():
            logs.append(f"{metric_name}={np.mean(values):.6f}")
        logger.info("  ".join(logs))

    logger.info("Evaluation finished.")


if __name__ == "__main__":
    cfg = OmegaConf.load("/home/obaidah/point-e/point_e/config.yaml")
    main(cfg)
