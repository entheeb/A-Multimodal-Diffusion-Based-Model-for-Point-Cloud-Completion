import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
from collections import defaultdict

'''Class names:
 - airplane
 - bed
 - bench
 - bottle
 - car
 - chair
 - desk
 - dresser
 - guitar
 - monitor
 - night_stand
 - piano
 - sofa
 - table
 - tent
 - toilet'''

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
        o3d.io.write_point_cloud(os.path.join(out_dir, f"target_{i + 1}.ply"), pcd)


def check_min_max_values_per_class(h5_path):
    with h5py.File(h5_path, 'r') as f:
        for class_name in f.keys():
            pointcloud_min, pointcloud_max = np.inf, -np.inf
            distance_min, distance_max = np.inf, -np.inf
            ground_truth_min, ground_truth_max = np.inf, -np.inf

            class_group = f[class_name]
            for instance_id in class_group.keys():
                instance_group = class_group[instance_id]

                if 'ground_truth' in instance_group:
                    gt = instance_group['ground_truth'][()]
                    ground_truth_min = min(ground_truth_min, gt.min())
                    ground_truth_max = max(ground_truth_max, gt.max())

                if 'partials' not in instance_group:
                    continue

                partials_group = instance_group['partials']
                for scan_name in partials_group.keys():
                    scan_group = partials_group[scan_name]

                    if 'pointcloud' in scan_group:
                        pc = scan_group['pointcloud'][()]
                        pointcloud_min = min(pointcloud_min, pc.min())
                        pointcloud_max = max(pointcloud_max, pc.max())

                    if 'distance' in scan_group:
                        dist = scan_group['distance'][()]
                        distance_min = min(distance_min, dist.min())
                        distance_max = max(distance_max, dist.max())

            print(f"\nClass: {class_name}")
            print(f"  Pointcloud → min: {pointcloud_min}, max: {pointcloud_max}")
            print(f"  Distance   → min: {distance_min}, max: {distance_max}")
            print(f"  Ground Truth → min: {ground_truth_min}, max: {ground_truth_max}")


def log_instances_with_incomplete_partials(h5_path, min_points=4096):
    with h5py.File(h5_path, 'r') as f:
        for class_name in f.keys():
            class_group = f[class_name]
            for instance_id in class_group.keys():
                instance_group = class_group[instance_id]

                if 'partials' not in instance_group:
                    continue

                partials_group = instance_group['partials']
                instance_has_small_scan = False

                for scan_name in partials_group.keys():
                    scan_group = partials_group[scan_name]
                    if 'pointcloud' in scan_group:
                        partial_pcd = scan_group['pointcloud'][()]
                        if partial_pcd.shape[0] < min_points:
                            instance_has_small_scan = True
                            break  # No need to check other scans for this instance

                if instance_has_small_scan:
                    print(f"[Instance with partial < {min_points}] {class_name}/{instance_id}")


def check_mean_variance(h5_path, batch_size=16, num_workers=0, skip_classes=["guitar"]):
    dataset = ModelnetDataset(h5_path=h5_path, skip_classes=skip_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    sum_partial, sum_sq_partial = 0.0, 0.0
    sum_depth, sum_sq_depth = 0.0, 0.0
    sum_gt, sum_sq_gt = 0.0, 0.0
    count_partial, count_depth, count_gt = 0, 0, 0

    for labels, partial_pcd, depth_maps, target_points in dataloader:
        sum_partial += partial_pcd.sum().item()
        sum_sq_partial += (partial_pcd ** 2).sum().item()
        count_partial += partial_pcd.numel()

        sum_depth += depth_maps.sum().item()
        sum_sq_depth += (depth_maps ** 2).sum().item()
        count_depth += depth_maps.numel()

        sum_gt += target_points.sum().item()
        sum_sq_gt += (target_points ** 2).sum().item()
        count_gt += target_points.numel()

    mean_partial = sum_partial / count_partial
    var_partial = (sum_sq_partial / count_partial) - (mean_partial ** 2)

    mean_depth = sum_depth / count_depth
    var_depth = (sum_sq_depth / count_depth) - (mean_depth ** 2)

    mean_gt = sum_gt / count_gt
    var_gt = (sum_sq_gt / count_gt) - (mean_gt ** 2)

    print(f"Partial PCD → mean: {mean_partial:.6f}, variance: {var_partial:.6f}")
    print(f"Depth Maps  → mean: {mean_depth:.6f}, variance: {var_depth:.6f}")
    print(f"Ground Truth→ mean: {mean_gt:.6f}, variance: {var_gt:.6f}")

def check_min_max_with_dataloader(h5_path, batch_size=8, num_workers=0, skip_classes=["guitar"]):
    dataset = ModelnetDatasetTest(h5_path=h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    pointcloud_min, pointcloud_max = float('inf'), float('-inf')
    depth_min, depth_max = float('inf'), float('-inf')
    gt_min, gt_max = float('inf'), float('-inf')

    for labels, partial_pcd, depth_maps, viewpoint, target_points in dataloader:
        pointcloud_min = min(pointcloud_min, partial_pcd.min().item())
        pointcloud_max = max(pointcloud_max, partial_pcd.max().item())

        depth_min = min(depth_min, depth_maps.min().item())
        depth_max = max(depth_max, depth_maps.max().item())

        gt_min = min(gt_min, target_points.min().item())
        gt_max = max(gt_max, target_points.max().item())

    print(f"Pointcloud → min: {pointcloud_min}, max: {pointcloud_max}")
    print(f"Depth Maps → min: {depth_min}, max: {depth_max}")
    print(f"Ground Truth → min: {gt_min}, max: {gt_max}")

def check_partial_scan_count_and_integrity(h5_path, expected_scans=36):
    with h5py.File(h5_path, 'r') as f:
        for class_name in f.keys():
            class_group = f[class_name]
            for instance_id in class_group.keys():
                instance_group = class_group[instance_id]
                if 'partials' not in instance_group:
                    print(f"[Missing partials] {class_name}/{instance_id}")
                    continue

                partials_group = instance_group['partials']
                scan_count = len(partials_group)

                if scan_count != expected_scans:
                    print(f"[Scan count mismatch] {class_name}/{instance_id} → {scan_count} scans (expected {expected_scans})")

                for scan_name in partials_group.keys():
                    scan_group = partials_group[scan_name]
                    if 'pointcloud' not in scan_group or 'distance' not in scan_group:
                        print(f"[Missing data] {class_name}/{instance_id}/{scan_name} → pointcloud or distance missing")

    print(" Check complete.")

def get_class_info(h5_path):
    with h5py.File(h5_path, 'r') as f:
        class_names = list(f.keys())
        print(f"Total classes: {len(class_names)}")
        print("Class names:")
        for name in sorted(class_names):
            print(f" - {name}")


def check_incomplete_instances(h5_path):
    with h5py.File(h5_path, 'r') as h5f:
        for class_name in h5f.keys():
            class_group = h5f[class_name]
            for instance_id in class_group.keys():
                instance_group = class_group[instance_id]

                if 'ground_truth' not in instance_group:
                    print(f"[Missing] ground_truth in {class_name}/{instance_id}")

                if 'class_label' not in instance_group:
                    print(f"[Missing] class_label in {class_name}/{instance_id}")

                if 'partials' not in instance_group or len(instance_group['partials']) == 0:
                    print(f"[Missing] partial scans in {class_name}/{instance_id}")

                else:
                    for scan_name in instance_group['partials'].keys():
                        scan_group = instance_group['partials'][scan_name]
                        if 'pointcloud' not in scan_group or 'distance' not in scan_group:
                            print(f"[Incomplete scan] {class_name}/{instance_id}/{scan_name}")



class ModelnetDataset(Dataset):
    def __init__(self, h5_path="/home/obaidah/point-e/point_e/dataset/train_dataset_1024.h5", skip_classes=["dresser","table", "desk", "bed","chair"]):
        self.h5_path = h5_path
        self.samples = []

        # List of broken instance paths: format → "<class_name>/<instance_id>"
        self.skip_instances = {
        "car/car_0239",
        "car/car_0241",
        "chair/chair_0940",
        "desk/desk_0241",
        "dresser/dresser_0243",
        "dresser/dresser_0244",
        "dresser/dresser_0251",
        "guitar/guitar_0158",
        "guitar/guitar_0191",
        "guitar/guitar_0194",
        "guitar/guitar_0205",
        "guitar/guitar_0216",
        "airplane/airplane_0087",
        "airplane/airplane_0103",
        "airplane/airplane_0152",
        "airplane/airplane_0207",
        "airplane/airplane_0378",
        "airplane/airplane_0433",
        "airplane/airplane_0449",
        "airplane/airplane_0477",
        "airplane/airplane_0485",
        "airplane/airplane_0512", }

        # Viewpoint list: scan_0000 → viewpoints[0], scan_0001 → viewpoints[1], etc.
        self.viewpoints = torch.tensor([
            (1.0, 0.0, 0.25),
            (0.9659258262890683, 0.25881904510252074, 0.25),
            (0.8660254037844387, 0.49999999999999994, 0.25),
            (0.7071067811865476, 0.7071067811865475, 0.25),
            (0.5, 0.8660254037844386, 0.25),
            (0.25881904510252074, 0.9659258262890683, 0.25),
            (6.123233995736766e-17, 1.0, 0.25),
            (-0.25881904510252063, 0.9659258262890683, 0.25),
            (-0.5, 0.8660254037844387, 0.25),
            (-0.7071067811865475, 0.7071067811865476, 0.25),
            (-0.8660254037844387, 0.5, 0.25),
            (-0.9659258262890683, 0.25881904510252096, 0.25),
            (-1.0, 1.2246467991473532e-16, 0.25),
            (-0.9659258262890684, -0.25881904510252063, 0.25),
            (-0.8660254037844386, -0.5, 0.25),
            (-0.7071067811865477, -0.7071067811865475, 0.25),
            (-0.5, -0.8660254037844387, 0.25),
            (-0.2588190451025213, -0.9659258262890682, 0.25),
            (-1.8369701987210297e-16, -1.0, 0.25),
            (0.2588190451025203, -0.9659258262890684, 0.25),
            (0.4999999999999996, -0.8660254037844388, 0.25),
            (0.7071067811865474, -0.7071067811865477, 0.25),
            (0.8660254037844384, -0.5000000000000004, 0.25),
            (0.9659258262890682, -0.25881904510252157, 0.25),
            (1.0, -2.4492935982947064e-16, 0.25),
            (0.9659258262890684, 0.25881904510252035, 0.25),
            (1.0, 0.0, 0.0),
            (0.777778, 0.0, 0.628539),
            (0.555556, 0.0, 0.831211),
            (0.333333, 0.0, 0.942809),
            (0.111111, 0.0, 0.993807),
            (-0.111111, 0.0, 0.993807),
            (-0.333333, 0.0, 0.942809),
            (-0.555556, 0.0, 0.831211),
            (-0.777778, 0.0, 0.628539),
            (-1.0, 0.0, 0.0),
        ], dtype=torch.float32)
        self.viewpoints = self.viewpoints[:, [0, 2, 1]]

        with h5py.File(h5_path, 'r') as f:
            if skip_classes is not None:
                class_names = [name for name in f.keys() if name not in skip_classes]
                class_names.sort()
                self.class_to_new_label = {cls: idx for idx, cls in enumerate(class_names)}
            else:
                self.class_to_new_label = {cls: idx for idx, cls in enumerate(f.keys())}


            for class_name in f.keys():
                if skip_classes and class_name in skip_classes:
                    continue

                class_group = f[class_name]
                for instance_id in class_group.keys():
                    instance_key = f"{class_name}/{instance_id}"
                    if instance_key in self.skip_instances:
                        continue

                    instance_group = class_group[instance_id]
                    partials_group = instance_group['partials']

                    for scan_name in partials_group.keys():
                        scan_idx = int(scan_name.split("_")[-1])  # Extract 0000 → 0

                        self.samples.append({
                            'original_class_name': class_name,
                            'partial_path': f"{class_name}/{instance_id}/partials/{scan_name}/pointcloud",
                            'depth_path': f"{class_name}/{instance_id}/partials/{scan_name}/distance",
                            'target_path': f"{class_name}/{instance_id}/ground_truth",
                            'viewpoint_idx': scan_idx
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            partial_pcd = f[sample['partial_path']][()]
            depth_maps = f[sample['depth_path']][()]
            target_points = f[sample['target_path']][()]

        # Convert to torch tensors
        partial_pcd = torch.from_numpy(partial_pcd).float().clamp(-0.5, 0.5)
        depth_maps = torch.from_numpy(depth_maps).float() / 255.0
        target_points = torch.from_numpy(target_points).float() * 0.01
        target_points = target_points.clamp(-0.5, 0.5)
        target_points = target_points[torch.randperm(target_points.shape[0])]

        class_name = sample['original_class_name']
        label_idx = torch.tensor(self.class_to_new_label[class_name], dtype=torch.long)
        
        viewpoint = self.viewpoints[sample['viewpoint_idx']]


        return label_idx, partial_pcd, depth_maps, viewpoint, target_points 
    
    def count_samples_per_class(self):
        """
        Returns a dictionary with the number of total scan samples per class
        and the number of unique instances.
        """

        class_counts = defaultdict(lambda: {"scans": 0, "instances": set()})

        for sample in self.samples:
            cls = sample['original_class_name']
            # increment scan count
            class_counts[cls]["scans"] += 1
            # extract instance_id from path (class/instance/partials/scan/pointcloud)
            instance_id = sample['partial_path'].split("/")[1]
            class_counts[cls]["instances"].add(instance_id)

        # convert instance sets to counts
        result = {
            cls: {
                "total_scans": data["scans"],
                "unique_instances": len(data["instances"]),
                "avg_scans_per_instance": data["scans"] / len(data["instances"])
            }
            for cls, data in class_counts.items()
        }
        return result
    


class ModelnetDatasetTest(Dataset):
    def __init__(self, h5_path="/home/obaidah/point-e/point_e/dataset/test_dataset_1024.h5", skip_classes=["dresser","table", "desk", "bed","chair"]):
        self.h5_path = h5_path
        self.samples = []

        # List of broken instance paths: format → "<class_name>/<instance_id>"
        self.skip_instances = {}

        # Viewpoint list: scan_0000 → viewpoints[0], scan_0001 → viewpoints[1], etc.
        self.viewpoints = torch.tensor([
            (1.0, 0.0, 0.25),
            (0.9659258262890683, 0.25881904510252074, 0.25),
            (0.8660254037844387, 0.49999999999999994, 0.25),
            (0.7071067811865476, 0.7071067811865475, 0.25),
            (0.5, 0.8660254037844386, 0.25),
            (0.25881904510252074, 0.9659258262890683, 0.25),
            (6.123233995736766e-17, 1.0, 0.25),
            (-0.25881904510252063, 0.9659258262890683, 0.25),
            (-0.5, 0.8660254037844387, 0.25),
            (-0.7071067811865475, 0.7071067811865476, 0.25),
            (-0.8660254037844387, 0.5, 0.25),
            (-0.9659258262890683, 0.25881904510252096, 0.25),
            (-1.0, 1.2246467991473532e-16, 0.25),
            (-0.9659258262890684, -0.25881904510252063, 0.25),
            (-0.8660254037844386, -0.5, 0.25),
            (-0.7071067811865477, -0.7071067811865475, 0.25),
            (-0.5, -0.8660254037844387, 0.25),
            (-0.2588190451025213, -0.9659258262890682, 0.25),
            (-1.8369701987210297e-16, -1.0, 0.25),
            (0.2588190451025203, -0.9659258262890684, 0.25),
            (0.4999999999999996, -0.8660254037844388, 0.25),
            (0.7071067811865474, -0.7071067811865477, 0.25),
            (0.8660254037844384, -0.5000000000000004, 0.25),
            (0.9659258262890682, -0.25881904510252157, 0.25),
            (1.0, -2.4492935982947064e-16, 0.25),
            (0.9659258262890684, 0.25881904510252035, 0.25),
            (1.0, 0.0, 0.0),
            (0.777778, 0.0, 0.628539),
            (0.555556, 0.0, 0.831211),
            (0.333333, 0.0, 0.942809),
            (0.111111, 0.0, 0.993807),
            (-0.111111, 0.0, 0.993807),
            (-0.333333, 0.0, 0.942809),
            (-0.555556, 0.0, 0.831211),
            (-0.777778, 0.0, 0.628539),
            (-1.0, 0.0, 0.0),
        ], dtype=torch.float32)
        self.viewpoints = self.viewpoints[:, [0, 2, 1]]

        with h5py.File(h5_path, 'r') as f:
            if skip_classes is not None:
                class_names = [name for name in f.keys() if name not in skip_classes]
                class_names.sort()
                self.class_to_new_label = {cls: idx for idx, cls in enumerate(class_names)}
            else:
                self.class_to_new_label = {cls: idx for idx, cls in enumerate(f.keys())}


            for class_name in f.keys():
                if skip_classes and class_name in skip_classes:
                    continue

                class_group = f[class_name]
                for instance_id in class_group.keys():
                    instance_key = f"{class_name}/{instance_id}"
                    if instance_key in self.skip_instances:
                        continue

                    instance_group = class_group[instance_id]
                    partials_group = instance_group['partials']

                    for scan_name in partials_group.keys():
                        scan_idx = int(scan_name.split("_")[-1])  # Extract 0000 → 0

                        self.samples.append({
                            'original_class_name': class_name,
                            'partial_path': f"{class_name}/{instance_id}/partials/{scan_name}/pointcloud",
                            'depth_path': f"{class_name}/{instance_id}/partials/{scan_name}/distance",
                            'target_path': f"{class_name}/{instance_id}/ground_truth",
                            'viewpoint_idx': scan_idx
                        })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            partial_pcd = f[sample['partial_path']][()]
            depth_maps = f[sample['depth_path']][()]
            target_points = f[sample['target_path']][()]

        # Convert to torch tensors
        partial_pcd = torch.from_numpy(partial_pcd).float().clamp(-0.5, 0.5)
        depth_maps = torch.from_numpy(depth_maps).float() / 255.0
        target_points = torch.from_numpy(target_points).float() * 0.01
        target_points = target_points.clamp(-0.5, 0.5)
        target_points = target_points[torch.randperm(target_points.shape[0])]

        class_name = sample['original_class_name']
        label_idx = torch.tensor(self.class_to_new_label[class_name], dtype=torch.long)
        
        viewpoint = self.viewpoints[sample['viewpoint_idx']]


        return label_idx, partial_pcd, depth_maps, viewpoint, target_points
    
    def count_samples_per_class(self):
        """
        Returns a dictionary with the number of total scan samples per class
        and the number of unique instances.
        """

        class_counts = defaultdict(lambda: {"scans": 0, "instances": set()})

        for sample in self.samples:
            cls = sample['original_class_name']
            # increment scan count
            class_counts[cls]["scans"] += 1
            # extract instance_id from path (class/instance/partials/scan/pointcloud)
            instance_id = sample['partial_path'].split("/")[1]
            class_counts[cls]["instances"].add(instance_id)

        # convert instance sets to counts
        result = {
            cls: {
                "total_scans": data["scans"],
                "unique_instances": len(data["instances"]),
                "avg_scans_per_instance": data["scans"] / len(data["instances"])
            }
            for cls, data in class_counts.items()
        }
        return result
    


def save_instance_ground_truths(h5_path, skip_classes=["dresser","table", "desk", "bed","chair"], npz_output="modelnet_filtered_instances.npz", pt_output="modelnet_filtered_labels.pt"):
    all_gt = []
    all_labels = []

    with h5py.File(h5_path, 'r') as f:
        class_names = [name for name in f.keys() if name not in skip_classes]
        class_names.sort()  # Same sorting as in your dataset
        class_to_new_label = {cls: idx for idx, cls in enumerate(class_names)}

        print("Class to label mapping:")
        for cls, idx in class_to_new_label.items():
            print(f"  {cls} → {idx}")

        for class_name in f.keys():
            if class_name in skip_classes:
                continue

            class_group = f[class_name]
            for instance_id in class_group.keys():
                instance_group = class_group[instance_id]

                # Get ground truth once per instance
                gt = instance_group['ground_truth'][()]  # (1024, 3)
                gt = torch.from_numpy(gt).float() * 0.01
                gt = gt.clamp(-0.5, 0.5)
                #gt = gt[torch.randperm(gt.shape[0])]  # Optional: randomize point order

                all_gt.append(gt.numpy())
                all_labels.append(class_to_new_label[class_name])

    all_gt = np.stack(all_gt)         # Shape: (N_instances, 1024, 3)
    all_labels = torch.tensor(all_labels, dtype=torch.long)  # Shape: (N_instances,)

    print(f"\nTotal unique instances: {all_gt.shape[0]}")
    print(f"Ground truth tensor shape: {all_gt.shape}  (expected: N x 1024 x 3)")
    print(f"Labels tensor shape: {all_labels.shape}     (expected: N,)")

    # Save ground truth point clouds
    np.savez_compressed(npz_output, ground_truths=all_gt)
    print(f"Saved ground truths to: {npz_output}")

    # Save labels
    torch.save(all_labels, pt_output)
    print(f"Saved labels to: {pt_output}")

    

if __name__ == "__main__":

    '''save_instance_ground_truths(
    h5_path="/home/obaidah/point-e/point_e/dataset/test_dataset_1024.h5",
    skip_classes=["dresser", "table", "desk", "bed", "chair"],
    npz_output="modelnet_filtered_instances.npz",
    pt_output="modelnet_filtered_labels.pt")'''

    dataset = ModelnetDatasetTest()
    print(f"Total samples: {len(dataset)}")
    stats = dataset.count_samples_per_class()

    # pretty print (sorted by class name)
    for cls in sorted(stats):
        s = stats[cls]
        print(f"{cls:12s} | scans={s['total_scans']:4d} | instances={s['unique_instances']:3d} | avg={s['avg_scans_per_instance']:.2f}")

    # or totals
    total_scans = sum(s["total_scans"] for s in stats.values())
    total_instances = sum(s["unique_instances"] for s in stats.values())
    print(f"\nTOTAL -> scans={total_scans}, unique_instances={total_instances}, "
        f"global_avg_scans_per_instance={total_scans / total_instances:.2f}")
    '''train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in train_loader:
        labels, partials, depths, viewpoint, targets = batch
        print(f"Labels: {labels.shape}, Partials: {partials.shape}, Depths: {depths.shape}, Targets: {targets.shape}, Viewpoints: {viewpoint.shape}")
        save_target_point_clouds(targets, out_dir="./target_pcds_modelnet", colors=None)
        save_target_point_clouds(partials, out_dir="./partial_pcds_modelnet", colors=None)
        break  # Just to test the first batch'''


    #check_incomplete_instances("/home/obaidah/point-e/point_e/dataset/test_dataset.h5")

    #get_class_info("/home/obaidah/point-e/point_e/dataset/train_dataset.h5")

    #check_partial_scan_count_and_integrity("/home/obaidah/point-e/point_e/dataset/train_dataset.h5", expected_scans=36)

    #check_min_max_values_per_class("/home/obaidah/point-e/point_e/dataset/train_dataset.h5")

    '''check_min_max_with_dataloader(
    h5_path="/home/obaidah/point-e/point_e/dataset/test_dataset_1024.h5",
    batch_size=32,
    num_workers=4
)'''

    #log_instances_with_incomplete_partials("/home/obaidah/point-e/point_e/dataset/test_dataset.h5", min_points=4096)
    '''check_mean_variance(
    h5_path="/home/obaidah/point-e/point_e/dataset/train_dataset.h5",
    batch_size=32,
    num_workers=4
     )'''