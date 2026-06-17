import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


###### AV1M ######

class AV1M_trainval_dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(self.csv_root_path, f"{self.split}_labels.csv"))
        self.feats_dir = os.path.join(self.root_path, self.split)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npz"), allow_pickle=True)
        label = int(row["label"])

        video = feats['visual']
        audio = feats['audio']

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, row["path"][:-4] + ".npz"  # video, audio, label, path


class AV1M_test_dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.csv_root_path = self.config["csv_root_path"]
        self.root_path = os.path.join(config["root_path"], "test_features")

        self.paths = np.load(os.path.join(self.root_path, "paths.npy"), allow_pickle=True)
        self.audio_feats = np.load(os.path.join(self.root_path, "audio.npy"), allow_pickle=True)
        self.video_feats = np.load(os.path.join(self.root_path, "video.npy"), allow_pickle=True)

        self.labels = self._get_labels()
    
    def _get_labels(self):
        df = pd.read_csv(os.path.join(self.csv_root_path, "test_labels.csv"))
        labels = {}
        for path in self.paths:
            row = df.loc[df['path'] == path]
            if len(row.index) != 1:
                raise ValueError("Multiple or no entries in test_labels.csv for a single path!")
            row = row.iloc[0]
            labels[path] = int(row['label'])

        return labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        video = self.video_feats[idx]
        audio = self.audio_feats[idx]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        label = self.labels[path]

        return torch.tensor(video), torch.tensor(audio), label, path  # video, audio, label, path


###### ######

###### AVLips ######

class AVLips_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_path = config["root_path"]

        dir_path_real = os.path.join(self.root_path, "0_real")
        dir_path_fake = os.path.join(self.root_path, "1_fake")

        paths_real = np.load(os.path.join(dir_path_real, "paths.npy"), allow_pickle=True)
        paths_fake = np.load(os.path.join(dir_path_fake, "paths.npy"), allow_pickle=True)

        self.labels = np.concatenate((np.zeros(paths_real.shape[0]), np.ones(paths_fake.shape[0]))).astype(np.int32)
        self.paths = np.concatenate((paths_real, paths_fake))
        self.audio_feats = np.concatenate((
            np.load(os.path.join(dir_path_real, "audio.npy"), allow_pickle=True),
            np.load(os.path.join(dir_path_fake, "audio.npy"), allow_pickle=True),
        ))
        self.video_feats = np.concatenate((
            np.load(os.path.join(dir_path_real, "video.npy"), allow_pickle=True),
            np.load(os.path.join(dir_path_fake, "video.npy"), allow_pickle=True),
        ))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video = self.video_feats[idx]
        audio = self.audio_feats[idx]
        path = self.paths[idx]
        label = self.labels[idx]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, path  # video, audio, label, path


###### ######

###### FakeAVCeleb ######

class FakeAVCeleb_Dataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        labels = pd.read_csv(os.path.join(self.csv_root_path, f"{split}_split.csv"))

        self.videos, self.audios, self.paths = np.array([]), np.array([]), np.array([])

        for folder_name in os.listdir(self.root_path):
            vids = np.load(os.path.join(self.root_path, folder_name, "video.npy"), allow_pickle=True)
            self.videos = np.concatenate((self.videos, vids))

            auds = np.load(os.path.join(self.root_path, folder_name, "audio.npy"), allow_pickle=True)
            self.audios = np.concatenate((self.audios, auds))

            ps = np.load(os.path.join(self.root_path, folder_name, "paths.npy"), allow_pickle=True)
            self.paths = np.concatenate((self.paths, ps))

        self.useful_data = []
        for idx in labels.index:
            row = labels.iloc[idx]
            path = row['full_path'].replace("FakeAVCeleb/", "")
            label = int(row['category'] != 'A')

            for id_path in range(len(self.paths)):
                if self.paths[id_path] == path:
                    self.useful_data.append((id_path, label))
                    break

    def __len__(self):
        return len(self.useful_data)

    def __getitem__(self, idx):
        id_path, label = self.useful_data[idx]
        video = self.videos[id_path]
        audio = self.audios[id_path]
        path = self.paths[id_path]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, path  # video, audio, label, path


###### ######

###### FakeAVCeleb NPZ (per-clip, cross-dataset eval) ######

class FakeAVCeleb_NPZ_Dataset(Dataset):
    """
    Per-clip NPZ dataset for FakeAVCeleb cross-dataset evaluation.

    Reads directly from deepfake_feature_extraction.py output:
      root_path/{RealVideo-RealAudio,RealVideo-FakeAudio,...}/.../clip.npz

    Label: 0 = Real (RealVideo-RealAudio), 1 = Fake (all other categories).

    Optionally filtered to a split CSV with columns source, category, full_path
    (from avh_sup/csv_metadata/favc/{train,val,test}_split.csv).
    """

    REAL_CATEGORY = "RealVideo-RealAudio"

    def __init__(self, config, split=None):
        self.config = config
        self.root_path = config["root_path"]        # e.g. data/favc_features/
        self.apply_l2 = config.get("apply_l2", False)

        # Optional: restrict to clips listed in a split CSV
        allowed = None
        if split is not None and "csv_root_path" in config:
            csv_path = os.path.join(config["csv_root_path"], f"{split}_split.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Extract video IDs and store with directory for prefix matching
                # Format: (directory_path, base_filename)
                allowed = set()
                for _, row in df.iterrows():
                    # Convert FakeAVCeleb/Category/.../id/00099.mp4 to Category/.../id/00099
                    base_path = row["full_path"].replace("FakeAVCeleb/", "").replace(".mp4", "")
                    # Split into directory and filename
                    parts = base_path.rsplit('/', 1)
                    if len(parts) == 2:
                        dir_path, filename = parts
                        allowed.add((dir_path, filename))
                    else:
                        # Fallback for edge cases
                        allowed.add(("", base_path))

        # Walk the feature directory
        self.items = []  # (abs_path, rel_path, label, category)
        if not os.path.isdir(self.root_path):
            raise FileNotFoundError(
                f"FakeAVCeleb features directory not found: {self.root_path}\n"
                "Run favc_extract_features.slurm first."
            )
        for cat in sorted(os.listdir(self.root_path)):
            cat_dir = os.path.join(self.root_path, cat)
            if not os.path.isdir(cat_dir):
                continue
            label = 0 if cat == self.REAL_CATEGORY else 1
            for dirpath, _, filenames in os.walk(cat_dir):
                for fname in sorted(filenames):
                    if not fname.endswith(".npz"):
                        continue
                    abs_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(abs_path, self.root_path)
                    
                    # Match against split CSV using flexible prefix matching
                    if allowed is not None:
                        # Get base path: Category/.../id/filename (strip .npz)
                        base_path = rel_path.replace(".npz", "")
                        
                        # Split into directory and filename
                        parts = base_path.rsplit('/', 1)
                        if len(parts) == 2:
                            dir_path, full_filename = parts
                            # Extract base filename (before first underscore if it exists multiple parts)
                            # e.g., "00130_id00173_UHoMXLSjlDo" -> "00130"
                            # or "00130_fake" -> "00130_fake" (keep simple suffixes)
                            base_filename = full_filename.split('_')[0]
                            
                            # Check if this matches any allowed entry
                            matched = False
                            for allowed_dir, allowed_name in allowed:
                                if dir_path == allowed_dir:
                                    # Check if filenames match (prefix or exact)
                                    if allowed_name == full_filename or \
                                       allowed_name == base_filename or \
                                       full_filename.startswith(allowed_name + '_'):
                                        matched = True
                                        break
                            
                            if not matched:
                                continue
                        else:
                            # Fallback: skip if can't parse
                            continue
                    
                    self.items.append((abs_path, rel_path, label, cat))

        n_real = sum(1 for _, _, l, _ in self.items if l == 0)
        n_fake = sum(1 for _, _, l, _ in self.items if l == 1)
        print(
            f"FakeAVCeleb_NPZ_Dataset [{split}]: {len(self.items)} clips  "
            f"(real={n_real}, fake={n_fake})",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        abs_path, rel_path, label, _ = self.items[idx]
        feats = np.load(abs_path, allow_pickle=True)
        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)

        if self.apply_l2:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True) + 1e-8)

        return torch.tensor(video), torch.tensor(audio), label, rel_path

    def categories(self):
        """Return list of (rel_path, label, category) for per-category metrics."""
        return [(rel, lbl, cat) for _, rel, lbl, cat in self.items]

###### ######

###### AVLips NPZ (per-clip, cross-dataset eval) ######

class AVLips_NPZ_Dataset(Dataset):
    """
    Per-clip NPZ dataset for AVLips cross-dataset evaluation.

    Reads directly from deepfake_feature_extraction.py AVLips output:
      root_path/0_real/*.npz  → label 0 (real)
      root_path/1_fake/*.npz  → label 1 (fake)

    Expected config keys:
      root_path   — path to avlips_features/ directory
      apply_l2    — (optional bool) L2-normalise features
    """

    CLASSES = {"0_real": 0, "1_fake": 1}

    def __init__(self, config):
        self.config = config
        self.root_path = config["root_path"]
        self.apply_l2 = config.get("apply_l2", False)

        if not os.path.isdir(self.root_path):
            raise FileNotFoundError(
                f"AVLips features directory not found: {self.root_path}\n"
                "Run extract_features_avlips.slurm first."
            )

        self.items = []  # (abs_path, rel_path, label)
        for cls, label in sorted(self.CLASSES.items()):
            cls_dir = os.path.join(self.root_path, cls)
            if not os.path.isdir(cls_dir):
                print(f"[AVLips_NPZ_Dataset] WARNING: {cls_dir} not found, skipping.")
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.endswith(".npz"):
                    continue
                abs_path = os.path.join(cls_dir, fname)
                rel_path = os.path.join(cls, fname)
                self.items.append((abs_path, rel_path, label))

        n_real = sum(1 for _, _, l in self.items if l == 0)
        n_fake = sum(1 for _, _, l in self.items if l == 1)
        print(
            f"AVLips_NPZ_Dataset: {len(self.items)} clips "
            f"(real={n_real}, fake={n_fake})",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        abs_path, rel_path, label = self.items[idx]
        feats = np.load(abs_path, allow_pickle=True)
        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)

        if self.apply_l2:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True) + 1e-8)

        return torch.tensor(video), torch.tensor(audio), label, rel_path

###### ######

def load_data(config, test=False):
    if test:
        if config["name"] == "AV1M":
            test_ds = AV1M_test_dataset(config)
        elif config["name"] == "AVLips":
            test_ds = AVLips_Dataset(config)
        elif config["name"] == "FAVC":
            test_ds = FakeAVCeleb_Dataset(config, split="test")
        elif config["name"] == "FAVC_NPZ":
            test_ds = FakeAVCeleb_NPZ_Dataset(config, split=config.get("split", "test"))
        elif config["name"] == "AVLIPS_NPZ":
            test_ds = AVLips_NPZ_Dataset(config)
        else:
            raise ValueError("Dataset name error. Expected: AV1M, AVLips, AVLIPS_NPZ, FAVC, FAVC_NPZ; Got: " + config["name"])

        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)

        return test_dl

    else:
        if config["name"] == "AV1M":
            train_ds = AV1M_trainval_dataset(config, split="train")
            val_ds = AV1M_trainval_dataset(config, split="val")
        elif config["name"] == "FAVC":
            train_ds = FakeAVCeleb_Dataset(config, split="train")
            val_ds = FakeAVCeleb_Dataset(config, split="val")
        else:
            raise ValueError("Dataset name error. Expected: AV1M, FAVC; Got: " + config["name"])

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=1)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=1)

        return train_dl, val_dl
