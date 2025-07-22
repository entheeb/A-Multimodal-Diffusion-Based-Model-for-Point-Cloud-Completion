'''if rank == 0:
                wandb.log({"loss": loss.item(), "step": global_step, "lr": scheduler.get_last_lr()[0]})
                progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1'''


'''# --- Condition Embedding Modules ---

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
    def __init__(
        self,
        num_points: int = 1024,
        num_latents: int = 256,
        cond_drop_prob: float = 0.1,
        input_channels: int = 3,
        output_channels: int = 3,
        latent_dim: int = 768,
        num_blocks: int = 6,
        num_compute_layers: int = 4,
        num_classes: int = 10,
        num_heads: int = 8,
        num_tokens_ppcd: int = 256,
        num_tokens_depth: int = 32,
        active_modalities: list = ["class", "view", "partial_pcd", "depth"],
        **kwargs,
    ):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.cond_drop_prob = cond_drop_prob
        self.active_modalities = active_modalities

        self.denoiser_backbone = Denoiser_backbone(
            input_channels=input_channels, output_channels=output_channels,
            num_x=num_points, num_z=num_latents, z_dim=latent_dim,
            num_blocks=num_blocks, num_compute_layers=num_compute_layers, num_heads=num_heads
        )

        # Encoders for active modalities
        self.encoders = nn.ModuleDict()
        self.token_type_ids = []

        if "class" in self.active_modalities:
            self.encoders["class"] = ClassEmbedding(num_classes=num_classes, embed_dim=latent_dim)
            self.token_type_ids.append((0, 1))  # (token_id, count)

        if "view" in self.active_modalities:
            self.encoders["view"] = ViewAngleEmbedding(input_dim=3, embed_dim=latent_dim)
            self.token_type_ids.append((1, 1))

        if "partial_pcd" in self.active_modalities:
            self.encoders["partial_pcd"] = PartialPointCloudEncoder(embed_dim=latent_dim, num_tokens=num_tokens_ppcd)
            self.token_type_ids.append((2, num_tokens_ppcd))

        if "depth" in self.active_modalities:
            self.encoders["depth"] = DepthMapEncoder(in_channels=1, embed_dim=latent_dim, num_tokens=num_tokens_depth)
            self.token_type_ids.append((3, num_tokens_depth))

        # Create token type embedding
        self.token_type_embeddings = nn.Embedding(4, latent_dim)  # Max 4 modalities (fixed IDs)
        nn.init.normal_(self.token_type_embeddings.weight, std=0.02)

        # Precompute token types template for active modalities
        token_type_list = []
        for token_id, count in self.token_type_ids:
            token_type_list += [token_id] * count
        self.register_buffer("token_types_template", torch.tensor(token_type_list, dtype=torch.long))
        self.total_tokens = len(token_type_list)

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
        return x_denoised, latent'''



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
from point_e.dataset.mvp_dataloader import MVP_CP


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

    dataloader, sampler = build_dataloader(cfg, rank, world_size, distributed)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=cfg.train.weight_decay)
    total_steps = int(len(dataloader) * cfg.train.epochs)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    #scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6, verbose=True)
    
    
    if rank == 0:
        best_checkpoints = []

    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        if distributed:
            sampler.set_epoch(epoch)

        #debug_batches = 0
        epoch_loss = 0.0
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        else:
            progress_bar = dataloader

        for batch in progress_bar:
            class_labels, partial_pcd, depth_maps, viewpoints, target = batch
            #class_labels, partial_pcd, target = batch
            target = target.permute(0, 2, 1).contiguous().to(device).float() # [B, C, N]

            t = torch.randint(0, cfg.diffusion.timesteps, (target.size(0),), device=target.device)

            noise = torch.randn_like(target)
            x_t = diffusion.q_sample(x_start=target, t=t, noise=noise)

            #class_labels = class_labels.to(device)
            partial_pcd = partial_pcd.to(device).float() 
            class_labels = None
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
                        partial_pcd=partial_pcd,
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

            '''if debug_batches >= 1500:
                break
            else:
                debug_batches += 1'''
        

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

                # Check if it's in the top 5
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
                logging.info(f"Best checkpoint so far: Epoch {best_epoch}, Loss {best_loss:.4f}")


    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    cfg = OmegaConf.load("/home/obaidah/point-e/point_e/config.yaml")
    main(cfg)
    print("Training completed.")
    wandb.finish()