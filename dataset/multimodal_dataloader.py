import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d 
import numpy as np

label_dict = {
    "airplane": 1,
    "cabinet": 2,
    "car": 3,
    "chair": 4,
    "lamp": 5,
    "sofa": 6,
    "table": 7,
    "watercraft": 8,
    "bed": 9,
    "bench": 10,
    "bookshelf": 11,
    "bus": 12,
    "guitar": 13,
    "motorbike": 14,
    "pistol": 15,
    "skateboard": 16,
}

def to_pcd(points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


class MultiModalDataset(Dataset):
    def __init__(self, h5_path="/shared/datasets/shapenet_multimodal/1024_scans_new.h5",
                  num_scans=154, dataset_name="car", depth_min=0, depth_max=255, viewpoints_max_abs=2.87765):
        super().__init__()
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.num_scans = num_scans
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.viewpoints_max_abs = viewpoints_max_abs
        self.skip_list = [("car", "car_974"),("car", "car_975"),("car", "car_976"),]
        print(f"Normalization stats computed: depth_min={self.depth_min}, depth_max={self.depth_max}, viewpoints_max_abs={self.viewpoints_max_abs}")
        self.load_dataset()

        if self.depth_min is None or self.depth_max is None or self.viewpoints_max_abs is None:
            self.compute_normalization_stats()


    def load_dataset(self):
        with h5py.File(self.h5_path, "r") as f:
            self.obj_types = list(f.keys())
            self.idx_list = []
            for obj_type in self.obj_types:
                obj_indices = [[obj_type, obj_id] for obj_id in f[obj_type].keys() if (obj_type, obj_id) not in self.skip_list]
                for obj_idx in obj_indices:
                    self.idx_list += [[obj_idx[0], obj_idx[1], i] for i in  np.linspace(0, 153, 40, dtype=int)]

    def compute_normalization_stats(self):
        print("Computing dataset normalization stats...")
        depth_min, depth_max = float("inf"), float("-inf")
        viewpoints_max_abs = float("-inf")

        with h5py.File(self.h5_path, "r") as f:
            for idx in range(len(self)):
                obj_type, obj_id, scan_idx = self.idx_list[idx]
                obj_data = f[obj_type][obj_id]
                depth = obj_data["depth_maps"][scan_idx][:]
                viewpoint = obj_data["viewpoints"][scan_idx][:]

                depth_min = min(depth_min, depth.min())
                depth_max = max(depth_max, depth.max())
                viewpoints_max_abs = max(viewpoints_max_abs, abs(viewpoint).max())

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.viewpoints_max_abs = viewpoints_max_abs
        print(f"Normalization stats computed: depth_min={self.depth_min}, depth_max={self.depth_max}, viewpoints_max_abs={self.viewpoints_max_abs}")


    def normalize_conditioning_inputs(self, depth_maps, viewpoints):
        depth_maps = (depth_maps - self.depth_min) / (self.depth_max - self.depth_min)
        viewpoints = viewpoints / self.viewpoints_max_abs
        return depth_maps, viewpoints
    
    def __len__(self):
        return  len(self.idx_list)

    def __getitem__(self, idx):
        obj_type, obj_id, scan_idx = self.idx_list[idx]
        #obj_id = f"car_{30}"
        #scan_idx += 2
        with h5py.File(self.h5_path, "r") as f:
            obj_data = f[obj_type][obj_id]
            partial_pcd = torch.from_numpy(obj_data["points"][scan_idx][:])
            depth_maps = torch.from_numpy(obj_data["depth_maps"][scan_idx][:]).float()
            #colors = torch.from_numpy(obj_data["colors"][scan_idx][:]) * 2.0 - 1.0  # Scale RGB from [0, 1] to [-1, 1]
            #target_color = torch.from_numpy(obj_data["target_colors"][:]) * 2.0 - 1.0  # Scale RGB from [0, 1] to [-1, 1]

            viewpoints = torch.from_numpy(obj_data["viewpoints"][scan_idx][:]).float()
            target_points = torch.from_numpy(obj_data["target_points"][:])
            target_points = target_points[torch.randperm(target_points.shape[0])]

            label_idx = torch.tensor(label_dict[obj_data.attrs.get("label", None)], dtype=torch.long)

    
        #partial_points_with_colors = torch.cat([partial_pcd, colors], dim=-1)  # Shape: (1024,6)
        #target_with_colors = torch.cat([target_points, target_color], dim=-1)  # Shape: (1024,6)

        # Normalize and standardize conditioning inputs
        depth_maps, viewpoints = self.normalize_conditioning_inputs(depth_maps, viewpoints)
        #depth_maps, viewpoints = torch.zeros_like(depth_maps), torch.zeros_like(viewpoints)  # For debugging, set to zero


        return label_idx, partial_pcd, depth_maps, viewpoints, target_points

if __name__ == "__main__":
    from tqdm import tqdm
    from collections import defaultdict
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # Load dataset
    dataset = MultiModalDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for (label_idx, partial_pcd, depth_map, viewpoint), target_with_colors in dataloader:
        #plt.figure(figsize=(10, 5))
        #plt.subplot(1, 2, 1)
        plt.imshow(depth_map[0].numpy())
        #plt.show(block=True)
        plt.savefig(f"depth_map_.png")
        break

    '''print("Inspecting first few samples...\n")

    for i, ((label_idx, partial_pcd, depth_map, viewpoint), target_with_colors) in enumerate(dataloader):
        print(f"--- Sample {i} ---")
        print(f"Label: {label_idx.item()}")

        # Shape checks
        print(f"partial_pcd:        {partial_pcd.shape}")       # [1, 1024, 6]
        print(f"target_with_colors: {target_with_colors.shape}")  # [1, 1024, 6]
        print(f"depth_map:          {depth_map.shape}")         # [1, 512, 512]
        print(f"viewpoint:          {viewpoint.shape}")         # [1, 3] (normalized)

        # Range checks
        print(f"partial_pcd (XYZ): min {partial_pcd[..., :3].min().item():.4f}, max {partial_pcd[..., :3].max().item():.4f}")
        print(f"partial_pcd (RGB): min {partial_pcd[..., 3:].min().item():.4f}, max {partial_pcd[..., 3:].max().item():.4f}")
        print(f"target_with_colors (RGB): min {target_with_colors[..., 3:].min().item():.4f}, max {target_with_colors[..., 3:].max().item():.4f}")
        print(f"depth_map: min {depth_map.min().item():.4f}, max {depth_map.max().item():.4f}")
        print(f"viewpoint: min {viewpoint.min().item():.4f}, max {viewpoint.max().item():.4f}")

        if i >= 4:
            break'''


    '''dataset = MultiModalDataset()
    stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_idx, (inputs, target_points) in enumerate(dataloader):
        label_idx, partial_pcd, depth_maps, viewpoints = inputs

        print(f"Batch {batch_idx}")
        print(f"  label_idx:    {label_idx.shape}")       # expected: [B]
        print(f"  partial_pcd:  {partial_pcd.shape}")     # expected: [B, 1024, 3]
        print(f"  depth_maps:   {depth_maps.shape}")      # expected: [B, 512, 512]
        print(f"  viewpoints:   {viewpoints.shape}")      # expected: [B, 3]
        print(f"  target_points:{target_points.shape}")   # expected: [B, N, 3] or similar

        if batch_idx == 10:  # only show 3 batches
            break

    for i in tqdm(range(len(dataset))):
        (label_idx, partial_pcd, depth_maps, viewpoints), target_points = dataset[i]
        
        stats["partial_pcd"]["min"] = min(stats["partial_pcd"]["min"], partial_pcd.min().item())
        stats["partial_pcd"]["max"] = max(stats["partial_pcd"]["max"], partial_pcd.max().item())

        stats["target_points"]["min"] = min(stats["target_points"]["min"], target_points.min().item())
        stats["target_points"]["max"] = max(stats["target_points"]["max"], target_points.max().item())

        #stats["depth_maps"]["min"] = min(stats["depth_maps"]["min"], depth_maps.min().item())
        #stats["depth_maps"]["max"] = max(stats["depth_maps"]["max"], depth_maps.max().item())

        #stats["viewpoints"]["min"] = min(stats["viewpoints"]["min"], viewpoints.min().item())
        #stats["viewpoints"]["max"] = max(stats["viewpoints"]["max"], viewpoints.max().item())

        #label_val = label_idx.item()
        #stats["label_idx"]["min"] = min(stats["label_idx"]["min"], label_val)
        #stats["label_idx"]["max"] = max(stats["label_idx"]["max"], label_val)

    print("\n==== Dataset Ranges After Normalization ====")
    for key, val in stats.items():
        print(f"{key}: min = {val['min']}, max = {val['max']}")
    

    dataset = MultiModalDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Accumulators
    def make_accumulator(dim):
        return {
            "sum": np.zeros(dim),
            "sum_sq": np.zeros(dim),
            "min": np.full(dim, np.inf),
            "max": np.full(dim, -np.inf),
            "count": 0,
        }

    stats = {
        "partial_pcd": make_accumulator(3),
        "target_points": make_accumulator(3),
        "viewpoints": make_accumulator(3),
        "depth_maps": {"sum": 0.0, "sum_sq": 0.0, "min": np.inf, "max": -np.inf, "count": 0},
        "label_idx": [],
    }

    print("Collecting stats from dataset...")

    for (label_idx, partial_pcd, depth_maps, viewpoints), target_points in tqdm(dataloader):
        # Convert to numpy
        partial_np = partial_pcd.view(-1, 3).numpy()
        target_np = target_points.view(-1, 3).numpy()
        view_np = viewpoints.view(-1).numpy()
        depth_np = depth_maps.view(-1).numpy()

        # Partial PCD
        stats["partial_pcd"]["sum"] += partial_np.sum(axis=0)
        stats["partial_pcd"]["sum_sq"] += (partial_np ** 2).sum(axis=0)
        stats["partial_pcd"]["min"] = np.minimum(stats["partial_pcd"]["min"], partial_np.min(axis=0))
        stats["partial_pcd"]["max"] = np.maximum(stats["partial_pcd"]["max"], partial_np.max(axis=0))
        stats["partial_pcd"]["count"] += partial_np.shape[0]

        # Target PCD
        stats["target_points"]["sum"] += target_np.sum(axis=0)
        stats["target_points"]["sum_sq"] += (target_np ** 2).sum(axis=0)
        stats["target_points"]["min"] = np.minimum(stats["target_points"]["min"], target_np.min(axis=0))
        stats["target_points"]["max"] = np.maximum(stats["target_points"]["max"], target_np.max(axis=0))
        stats["target_points"]["count"] += target_np.shape[0]

        # Viewpoints
        stats["viewpoints"]["sum"] += view_np
        stats["viewpoints"]["sum_sq"] += view_np ** 2
        stats["viewpoints"]["min"] = np.minimum(stats["viewpoints"]["min"], view_np)
        stats["viewpoints"]["max"] = np.maximum(stats["viewpoints"]["max"], view_np)
        stats["viewpoints"]["count"] += 1

        # Depth maps
        stats["depth_maps"]["sum"] += depth_np.sum()
        stats["depth_maps"]["sum_sq"] += (depth_np ** 2).sum()
        stats["depth_maps"]["min"] = min(stats["depth_maps"]["min"], depth_np.min())
        stats["depth_maps"]["max"] = max(stats["depth_maps"]["max"], depth_np.max())
        stats["depth_maps"]["count"] += depth_np.size

        # Labels
        stats["label_idx"].append(label_idx.item())

    def summarize_vector(name, acc):
        mean = acc["sum"] / acc["count"]
        var = (acc["sum_sq"] / acc["count"]) - mean**2
        std = np.sqrt(var)
        print(f" {name}:\n  Mean: {mean}\n  Std:  {std}\n  Min:  {acc['min']}\n  Max:  {acc['max']}\n")

    def summarize_scalar(name, acc):
        mean = acc["sum"] / acc["count"]
        var = (acc["sum_sq"] / acc["count"]) - mean**2
        std = np.sqrt(var)
        print(f" {name}:\n  Mean: {mean:.6f}\n  Std:  {std:.6f}\n  Min:  {acc['min']}\n  Max:  {acc['max']}\n")

    summarize_vector("Partial Point Cloud (XYZ)", stats["partial_pcd"])
    summarize_vector("Target Point Cloud (XYZ)", stats["target_points"])
    summarize_vector("Viewpoint Angles (XYZ)", stats["viewpoints"])
    summarize_scalar("Depth Maps (flattened)", stats["depth_maps"])

    label_arr = np.array(stats["label_idx"])
    print(f" Class Labels:\n  Min: {label_arr.min()}, Max: {label_arr.max()}, Unique: {sorted(set(label_arr))}")'''


    '''# Pick an example object
    example_obj_type = "car"    # change as needed
    example_obj_id = "car_500"    # change as needed

    with h5py.File("/shared/datasets/shapenet_multimodal/1024_scans_new.h5", "r") as f:
        obj = f[example_obj_type][example_obj_id]
        viewpoints = np.stack([obj["viewpoints"][i][:] for i in range(154)], axis=0)  # (154, 3)
    viewpoints = viewpoints / (2.877651214 + 1e-8)  # Normalize viewpoints

    print("First 30 viewpoints (as rows):")
    print(viewpoints[:30])

    # Plot and SAVE
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(viewpoints[:,0], viewpoints[:,1], viewpoints[:,2])
    ax.set_title(f"Viewpoints for {example_obj_id}")
    plt.tight_layout()
    plt.savefig("viewpoints_car_2.png")    # <-- Save the plot
    plt.close(fig)                        # <-- Clean up

    print("Saved plot as viewpoints_car_1.png")'''
