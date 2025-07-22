import torch
from torch.utils.data import DataLoader
import open3d as o3d 
import os
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import h5py
from .point_ops import fps


class MVP_CP(data.Dataset):
    def __init__(self, prefix="train", n_samples=None, device=torch.device("cuda"), limit=None):
        if prefix == "train":
            self.file_path = "/shared/datasets/mvp_data/MVP_Train_CP.h5"
        elif prefix == "val":
            self.file_path = "/shared/datasets/mvp_data/MVP_Test_CP.h5"
        elif prefix == "test":
            self.file_path = "/shared/datasets/mvp_data/MVP_Test_CP.h5"
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        self.limit = limit
        self.prefix = prefix
        self.n_samples = 2048 if n_samples is None else n_samples

        input_file = h5py.File(self.file_path, "r")
        self.input_data = np.array(input_file["incomplete_pcds"][()])
        self.input_data = self.input_data[:limit] if limit else self.input_data
        # print(self.input_data.shape)

        if prefix != "test":
            self.gt_data = np.array(input_file["complete_pcds"][()])
            self.labels = np.array(input_file["labels"][()])
            if self.n_samples < 2048:
                b = self.gt_data.copy()
                self.gt_data = fps(torch.from_numpy(b).to(device), n_samples=self.n_samples)
                self.gt_data = self.gt_data.to("cpu", dtype=torch.float16)
                print(self.gt_data.shape)

        if self.n_samples < 2048:
            a = self.input_data.copy()
            self.input_data = fps(torch.from_numpy(a).to(device), n_samples=self.n_samples)
            self.input_data = self.input_data.to("cpu", dtype=torch.float16)
            print(self.input_data.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = self.input_data[index]
        #partial = partial[torch.randperm(partial.shape[0])]

        if self.prefix != "test":
            
            complete = self.gt_data[index // 26]
            #complete = self.gt_data[index]
            complete = complete[torch.randperm(complete.shape[0])]

            label = self.labels[index]
            return torch.tensor(label, dtype=torch.long), partial, complete
        else:
            return partial



def save_list_as_txt(text_list, fp):
    with open(f"{fp}.txt", "w") as f:
        for line in text_list:
            f.write(f"{line}\n")

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


if __name__ == "__main__":
    dataset = MVP_CP(prefix="train", n_samples=1024, limit=None)
    dataloader = DataLoader(dataset, batch_size=26, shuffle=False, num_workers=4)

    for i, (label, partial, complete) in tqdm(enumerate(dataloader), total=len(dataloader)):
        print(f"Batch {i}:")
        print(f"Label: {label.shape}, Partial: {partial.shape}, Complete: {complete.shape}")
        # Save the first batch for demonstration
        save_target_point_clouds(partial, out_dir="/home/obaidah/point-e/point_e/MVP/partial_pcds", colors=None)
        save_target_point_clouds(complete, out_dir="/home/obaidah/point-e/point_e/MVP/complete_pcds", colors=None)
        if i == 0:  # Just to limit the output for demonstration
            break

    '''# Set config
    file_path = "/home/obaidah/point-e/point_e/dataset/MVP_Balanced_Train_CP.h5"
    n_samples = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load HDF5 file
    with h5py.File(file_path, "r") as f:
        partial_pcds = np.array(f["incomplete_pcds"][()])
        complete_pcds = np.array(f["complete_pcds"][()])
        labels = np.array(f["labels"][()])

    # Show number of scans and instances
    num_scans = partial_pcds.shape[0]
    num_instances = complete_pcds.shape[0]
    scans_per_instance = num_scans // num_instances

    print(f"Number of scans (partial point clouds): {num_scans}")
    print(f"Number of unique object instances: {num_instances}")
    print(f"Scans per instance: {scans_per_instance}")

    # Show number of classes
    unique_classes = np.unique(labels)
    print(f"Number of unique classes: {len(unique_classes)}")
    print(f"Class labels: {unique_classes}")

    # Apply FPS to partial and complete (downsample to 1024 points)
    print("Applying FPS to 1024 points per sample...")
    partial_tensor = torch.from_numpy(partial_pcds).to(device)
    complete_tensor = torch.from_numpy(complete_pcds).to(device)

    partial_1024 = fps(partial_tensor, n_samples=n_samples).to("cpu").numpy()
    complete_1024 = fps(complete_tensor, n_samples=n_samples).to("cpu").numpy()

    # Check range of values
    partial_min = partial_1024.min()
    partial_max = partial_1024.max()
    complete_min = complete_1024.min()
    complete_max = complete_1024.max()

    print(f"Partial point cloud value range: min={partial_min:.4f}, max={partial_max:.4f}")
    print(f"Complete point cloud value range: min={complete_min:.4f}, max={complete_max:.4f}")

    from collections import defaultdict

    # Calculate number of unique object instances
    num_instances = labels.shape[0] // 26  # 26 scans per instance
    labels_per_instance = labels.reshape(num_instances, 26)[:, 0]  # take first label of each group

    # Count number of instances per class
    class_counts = defaultdict(int)
    for label in labels_per_instance:
        class_counts[int(label)] += 1

    print(f"\nTotal unique object instances: {num_instances}")
    print("Number of unique instances per class:")
    for cls_id in sorted(class_counts):
        print(f"  Class {cls_id}: {class_counts[cls_id]} instances")'''
    
    '''import numpy as np
    import h5py
    from collections import defaultdict

    def load_dataset(path):
        with h5py.File(path, "r") as f:
            partial = np.array(f["incomplete_pcds"][()])
            complete = np.array(f["complete_pcds"][()])
            labels = np.array(f["labels"][()])
        return partial, complete, labels

    def group_by_class(partial, complete, labels):
        instances = defaultdict(list)
        num_instances = partial.shape[0] // 26
        labels_per_instance = labels.reshape(num_instances, 26)[:, 0]

        for i in range(num_instances):
            cls = int(labels_per_instance[i])
            start = i * 26
            end = start + 26
            instances[cls].append({
                "partial": partial[start:end],
                "complete": complete[i],
                "label": cls
            })
        return instances

    # Load datasets
    train_p, train_c, train_l = load_dataset("/shared/datasets/mvp_data/MVP_Train_CP.h5")
    test_p, test_c, test_l = load_dataset("/shared/datasets/mvp_data/MVP_Test_CP.h5")

    train_instances = group_by_class(train_p, train_c, train_l)
    test_instances = group_by_class(test_p, test_c, test_l)

    # Merge train + test, but prioritize train
    balanced_partial = []
    balanced_complete = []
    balanced_labels = []

    new_test_partial = []
    new_test_complete = []
    new_test_labels = []

    for cls in range(16):
        train_cls = train_instances[cls]
        test_cls = test_instances[cls]

        num_train = len(train_cls)
        num_needed = 150
        used_train = min(num_train, num_needed)
        needed_from_test = num_needed - used_train

        selected_train = train_cls[:used_train]
        selected_test = test_cls[:needed_from_test]
        remaining_test = test_cls[needed_from_test:]

        selected = selected_train + selected_test

        print(f"Class {cls}: using {len(selected_train)} from train, {len(selected_test)} from test → total {len(selected)}")
        print(f"Class {cls}: remaining {len(remaining_test)} instances in test for new test set")

        # Add to balanced dataset
        for item in selected:
            balanced_partial.append(item["partial"])
            balanced_complete.append(item["complete"])
            balanced_labels.extend([cls] * 26)

        # Add unused test to new test set
        for item in remaining_test:
            new_test_partial.append(item["partial"])
            new_test_complete.append(item["complete"])
            new_test_labels.extend([cls] * 26)

    # Stack arrays
    balanced_partial = np.concatenate(balanced_partial, axis=0)
    balanced_complete = np.stack(balanced_complete, axis=0)
    balanced_labels = np.array(balanced_labels)

    new_test_partial = np.concatenate(new_test_partial, axis=0)
    new_test_complete = np.stack(new_test_complete, axis=0)
    new_test_labels = np.array(new_test_labels)

    # Print summaries
    print(f"\n✅ Final balanced train set:")
    print(f"- Partial: {balanced_partial.shape}")
    print(f"- Complete: {balanced_complete.shape}")
    print(f"- Labels: {balanced_labels.shape}")

    print(f"\n✅ Final test set:")
    print(f"- Partial: {new_test_partial.shape}")
    print(f"- Complete: {new_test_complete.shape}")
    print(f"- Labels: {new_test_labels.shape}")

    # Save both sets
    with h5py.File("MVP_Balanced_Train_CP.h5", "w") as f:
        f.create_dataset("incomplete_pcds", data=balanced_partial)
        f.create_dataset("complete_pcds", data=balanced_complete)
        f.create_dataset("labels", data=balanced_labels)
        print("✅ Saved MVP_Balanced_Train_CP.h5")

    with h5py.File("MVP_Remaining_Test_CP.h5", "w") as f:
        f.create_dataset("incomplete_pcds", data=new_test_partial)
        f.create_dataset("complete_pcds", data=new_test_complete)
        f.create_dataset("labels", data=new_test_labels)
        print("✅ Saved MVP_Remaining_Test_CP.h5")'''
    
    '''import h5py
    import numpy as np
    from collections import defaultdict
    import hashlib

    def hash_point_cloud(pc):
        """Hash a point cloud array (Nx3) into a digest."""
        return hashlib.md5(pc.tobytes()).hexdigest()

    def check_duplicates(h5_path):
        with h5py.File(h5_path, "r") as f:
            complete = np.array(f["complete_pcds"][()])
            labels = np.array(f["labels"][()])

        num_instances = complete.shape[0]
        labels_per_instance = labels.reshape(num_instances, 26)[:, 0]

        class_hashes = defaultdict(set)
        duplicate_count = defaultdict(int)

        for i in range(num_instances):
            cls = int(labels_per_instance[i])
            pc = complete[i]
            h = hash_point_cloud(pc)
            if h in class_hashes[cls]:
                duplicate_count[cls] += 1
            else:
                class_hashes[cls].add(h)

        print(f"🔍 Duplicate check for: {h5_path}")
        for cls in sorted(class_hashes.keys()):
            total = len(class_hashes[cls]) + duplicate_count[cls]
            dup = duplicate_count[cls]
            print(f"  Class {cls}: {total} instances, {dup} duplicates")
        print("✅ Done.")

    # Run check on both balanced train and test files
    check_duplicates("/home/obaidah/point-e/point_e/dataset/MVP_Balanced_Train_CP.h5")
    check_duplicates("/home/obaidah/point-e/point_e/dataset/MVP_Remaining_Test_CP.h5")'''