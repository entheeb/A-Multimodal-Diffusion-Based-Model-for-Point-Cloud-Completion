import h5py
import torch
import numpy as np
from tqdm import tqdm
from pointnet.utils import farthest_point_sampling


def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1).repeat(1, idx.shape[1])
    new_points = points[batch_indices, idx, :]
    return new_points


def batch_fps_tensor(pcd_tensor, n_samples=1024):
    fps_idx = farthest_point_sampling(pcd_tensor, n_samples)
    sampled_pcds = index_points(pcd_tensor, fps_idx)
    return sampled_pcds


def create_downsampled_h5_full(original_h5_path, output_h5_path, n_samples=1024, device="cuda"):
    skip_instances = {
        "car/car_0239", "car/car_0241", "chair/chair_0940", "desk/desk_0241",
        "dresser/dresser_0243", "dresser/dresser_0244", "dresser/dresser_0251",
        "guitar/guitar_0158", "guitar/guitar_0191", "guitar/guitar_0194", "guitar/guitar_0205", "guitar/guitar_0216",
        "airplane/airplane_0087", "airplane/airplane_0103", "airplane/airplane_0152", "airplane/airplane_0207",
        "airplane/airplane_0378", "airplane/airplane_0433", "airplane/airplane_0449", "airplane/airplane_0477",
        "airplane/airplane_0485", "airplane/airplane_0512"
    }
    with h5py.File(original_h5_path, 'r') as src, h5py.File(output_h5_path, 'w') as dst:
        for class_name in tqdm(src.keys(), desc="Processing classes"):
            if class_name == "guitar":
                print(f"Skipping class: {class_name}")
                continue
            class_group_src = src[class_name]
            class_group_dst = dst.create_group(class_name)
            instance_ids = list(class_group_src.keys())
            
            # Preload all valid data for this class
            valid_instances = []
            gt_list = []
            partials_list = []
            meta_info = []
            for instance_id in instance_ids:
                instance_key = f"{class_name}/{instance_id}"
                if instance_key in skip_instances:
                    print(f"[Skipped instance] {instance_key} (in skip list)")
                    continue
                instance_group_src = class_group_src[instance_id]
                gt = instance_group_src['ground_truth'][()]
                if gt.shape[0] < 4096:
                    print(f"[Skipped instance] {instance_key} → ground truth shape: {gt.shape}")
                    continue
                partials_group_src = instance_group_src['partials']
                valid = True
                temp_partial_list = []
                temp_meta = []
                for scan_name in partials_group_src.keys():
                    pc = partials_group_src[scan_name]['pointcloud'][()]
                    if pc.shape[0] < 4096:
                        print(f"[Skipped instance] {instance_key} → scan {scan_name} shape: {pc.shape}")
                        valid = False
                        break
                    temp_partial_list.append(pc)
                    temp_meta.append(scan_name)
                if not valid:
                    continue
                valid_instances.append((instance_id, instance_group_src['class_label'][()], temp_meta))
                gt_list.append(gt)
                partials_list.append(temp_partial_list)
                meta_info.append(temp_meta)
            # Apply FPS on all gt together
            if gt_list:
                gt_tensor = torch.from_numpy(np.stack(gt_list)).float().to(device)
                sampled_gt_tensor = batch_fps_tensor(gt_tensor, n_samples).cpu().numpy()
                # Apply FPS on all partials together
                flat_partials = [pc for instance_partial in partials_list for pc in instance_partial]
                flat_partials_tensor = torch.from_numpy(np.stack(flat_partials)).float().to(device)
                sampled_partials_tensor = batch_fps_tensor(flat_partials_tensor, n_samples).cpu().numpy()
                # Write to H5
                idx = 0
                for (instance_id, class_label, instance_meta), sampled_gt in zip(valid_instances, sampled_gt_tensor):
                    instance_group_dst = class_group_dst.create_group(instance_id)
                    instance_group_dst.create_dataset('ground_truth', data=sampled_gt)
                    instance_group_dst.create_dataset('class_label', data=class_label)
                    partials_group_dst = instance_group_dst.create_group('partials')
                    for scan_name in instance_meta:
                        scan_group_dst = partials_group_dst.create_group(scan_name)
                        scan_group_dst.create_dataset('pointcloud', data=sampled_partials_tensor[idx])
                        distance = class_group_src[instance_id]['partials'][scan_name]['distance'][()]
                        scan_group_dst.create_dataset('distance', data=distance)
                        idx += 1
    print(f"✅ New H5 saved: {output_h5_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    create_downsampled_h5_full(
        original_h5_path="/home/obaidah/point-e/point_e/dataset/train_dataset.h5",
        output_h5_path="/home/obaidah/point-e/point_e/dataset/train_dataset_1024.h5",
        n_samples=1024,
        device=device
    )
