# train.py

import os
import math
import time
import datetime
import torch
import wandb
import random
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup


from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.model import TwoStreamDenoiser
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from point_e.dataset.multimodal_dataloader import MultiModalDataset
from point_e.dataset.modelnet_dataloader import ModelnetDataset
from point_e.dataset.mvp_dataloader import MVP_CP
from point_e.models.util import save_samples, save_target_point_clouds


# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_dataloader(cfg, rank=0, world_size=1, distributed=False):
    dataset = MultiModalDataset(h5_path=cfg.data.h5_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, sampler

def build_modelnet_dataloader(cfg, rank=0, world_size=1, distributed=False):
    dataset = ModelnetDataset(h5_path=cfg.data.h5_path,)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler

def build_dataloader_mvp(cfg, rank=0, world_size=1, distributed=False):
    dataset = MVP_CP(prefix="train", n_samples=cfg.model.num_points)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, sampler

# -------------------------
def main(cfg):
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ["RANK"]) if distributed else 0
    world_size = int(os.environ["WORLD_SIZE"]) if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if distributed:
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # -------------------------
    # Experiment folder & logging
    # -------------------------
    if rank == 0:
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')
        experiment_name = f"run_{timestamp}"
        experiment_dir = os.path.join(cfg.train.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        OmegaConf.save(cfg, os.path.join(experiment_dir, "config_used.yaml"))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logging.info(f"Starting experiment: {experiment_name}")

        wandb.init(
            project=cfg.wandb.project,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    set_seed(cfg.train.seed + rank)

    model = TwoStreamDenoiser(**cfg.model).to(device)

    if cfg.train.continue_training:
        checkpoint_path = cfg.train.load_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))


    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    steps = cfg.diffusion.timesteps
    schedule = cfg.diffusion.schedule
    betas = get_named_beta_schedule(schedule, steps)
    diffusion = GaussianDiffusion(**cfg.diffusion.gaussiandiffusion, betas=betas)

    dataloader, sampler = build_modelnet_dataloader(cfg, rank, world_size, distributed)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=cfg.train.weight_decay)
    total_steps = int(len(dataloader) * cfg.train.epochs )
    #last_epoch = int(len(dataloader) * cfg.train.epochs ) - 1
    #warmup_steps = int(0.05 * total_steps)
    #scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps, last_epoch=last_epoch)
    #scheduler = build_warm_cosine_scheduler(optimizer, total_steps, peak_lr=cfg.train.lr, warmup_ratio=0.05, lr_floor=0.01, last_epoch= last_epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    
    if rank == 0:
        best_checkpoints = []

    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        if distributed:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        else:
            progress_bar = dataloader

        for batch in progress_bar:
            class_labels, partial_pcd, depth_maps, viewpoints, target = batch
            target = target.permute(0, 2, 1).contiguous().to(device) # [B, C, N]

            t = torch.randint(0, cfg.diffusion.timesteps, (target.size(0),), device=target.device)

            noise = torch.randn_like(target)
            x_t = diffusion.q_sample(x_start=target, t=t, noise=noise)

            class_labels = class_labels.to(device)
            partial_pcd = partial_pcd.to(device)
            #class_labels = None
            #partial_pcd = None
            depth_maps = depth_maps.to(device).unsqueeze(1)
            viewpoints = viewpoints.to(device)

            # Self-conditioning logic:
            if random.random() < cfg.train.self_conditioning_prob:
                with torch.no_grad():
                    # simulate prev_latent using same batch input
                    _, prev_latent = model(
                        x_t, t, 
                        class_labels=class_labels, 
                        #partial_pcd=partial_pcd,
                        depth_maps=depth_maps,
                        viewpoints=viewpoints,
                        prev_latent=None
                    )
                prev_latent = prev_latent.detach()
            else:
                prev_latent = None
            # -----------------------------------------

            '''model_kwargs = {
                    "class_labels": class_labels,  
                    "partial_pcd": partial_pcd,
                    "prev_latent": prev_latent}'''


            model_kwargs = {
                    "class_labels": class_labels,
                    "viewpoints": viewpoints,  
                    "partial_pcd": partial_pcd,
                    "depth_maps": depth_maps,  
                    "prev_latent": prev_latent,
                }

            
            if epoch + 1 <= cfg.train.start_chamfer:
                loss_dict = diffusion.training_losses(
                    model,
                    target,
                    t,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    use_cd_xyz_loss=False,
                    use_cd_color_loss=False
                )
            else:
                loss_dict = diffusion.training_losses(
                    model,
                    target,
                    t,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    use_cd_xyz_loss=True,
                    use_cd_color_loss=False
                )


            loss = loss_dict["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()


            # Sync loss across all GPUs for logging
            if distributed:
                loss_clone = loss.clone().detach()
                dist.all_reduce(loss_clone, op=dist.ReduceOp.SUM)
                loss_clone /= world_size
                loss_to_log = loss_clone.item()
            else:
                loss_to_log = loss.item()
            

            epoch_loss += loss_to_log
            if rank == 0:               
                # Log overall loss as well
                wandb.log({
                    "loss": loss_to_log,
                    "step": global_step,
                    "lr":  scheduler.get_last_lr()[0],
                })
                progress_bar.set_postfix({"loss": loss_to_log})

            global_step += 1


        # -------------------------
        # Save checkpoint
        # -------------------------
        
        if rank == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            logging.info(f"Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f}")

            if (epoch + 1) % cfg.train.save_every == 0:
                # Save the current checkpoint
                ckpt_path = os.path.join(experiment_dir, f"model_epoch_{epoch + 1}.pt")
                torch.save(
                    model.module.state_dict() if distributed else model.state_dict(),
                    ckpt_path,
                )
                logging.info(f"Saved checkpoint to {ckpt_path}")

                '''# Check if it's in the top 5
                best_checkpoints.append((epoch + 1, avg_epoch_loss, ckpt_path))
                # Sort by loss (lower is better)
                best_checkpoints = sorted(best_checkpoints, key=lambda x: x[1])

                # If more than 5, remove the worst
                if len(best_checkpoints) > 6 and epoch + 1 < cfg.train.epochs:
                    worst_epoch, worst_loss, worst_path = best_checkpoints.pop(-1)
                    try:
                        os.remove(worst_path)
                        logging.info(f"Deleted checkpoint {worst_path} (loss={worst_loss:.4f})")
                    except OSError as e:
                        logging.warning(f"Could not delete {worst_path}: {e}")
                # Log the best current checkpoint loss
                best_epoch, best_loss, best_path = best_checkpoints[0]
                logging.info(f"Best checkpoint so far: Epoch {best_epoch}, Loss {best_loss:.4f}")'''

            if (epoch + 1) % cfg.train.sample_every == 0:
                # Run sampling on last batch
                    model.eval()
                    sampler_fn = PointCloudSampler(
                        device=device,
                        models=[model.module if distributed else model],
                        diffusions=[diffusion],
                        num_points=[1024],
                        aux_channels=[],
                        guidance_scale=[cfg.sample.guidance_scale],
                        clip_denoised=True,
                        use_karras=[cfg.sample.use_karras],
                        karras_steps=[cfg.sample.karras_steps],
                        sigma_min=[cfg.sample.sigma_min],
                        sigma_max=[cfg.sample.sigma_max],
                        s_churn=[cfg.sample.s_churn],
                    )

                    model_kwargs = {
                    "class_labels": class_labels,
                    "viewpoints": viewpoints,  
                    "partial_pcd": partial_pcd,
                    "depth_maps": depth_maps,  
                    }              

                    partial_pcd_output_dir = os.path.join(experiment_dir, f"partial_pcd_epoch_{epoch + 1}")
                    target_points_output_dir = os.path.join(experiment_dir, f"target_points_epoch_{epoch + 1}")
                    save_target_point_clouds(partial_pcd, partial_pcd_output_dir, prefix="partial_pcd")
                    save_target_point_clouds(target.permute(0, 2, 1).contiguous(), target_points_output_dir, prefix="target_points")

                    print("Sampling from last batch...")
                    all_samples = []
                    for sample in sampler_fn.sample_batch_progressive(target.size(0), model_kwargs, x_target=None):
                        all_samples.append(sample)
                    final_samples = all_samples[-1]

                    sample_output_dir = os.path.join(experiment_dir, f"samples_epoch_{epoch + 1}")
                    save_samples(final_samples, sample_output_dir)
                    print(f"Saved samples to {sample_output_dir}")


    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    cfg = OmegaConf.load("/home/obaidah/point-e/point_e/config.yaml")
    main(cfg)
    print("Training completed.")
    wandb.finish()
