import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class FMRIH5Dataset(Dataset):
    def __init__(self, h5_path, meta_path, desc_root, desc_max_len=64):
        self.h5_file = h5py.File(h5_path, "r")
        self.stimuli = self.h5_file["stimuli"]
        self.fmri = self.h5_file["fmri"]
        self.desc_root = desc_root
        self.desc_max_len = desc_max_len

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.subject_map = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}
        self.subject_labels = [self.subject_map[item["subject"]] for item in self.meta]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.stimuli[idx], dtype=torch.float32)
        y = torch.tensor(self.fmri[idx], dtype=torch.float32)
        subject_id = torch.tensor(self.subject_labels[idx], dtype=torch.long)

        episode = self.meta[idx]["episode"]
        if "movie10" in episode:
            desc_path = os.path.join(self.desc_root, f"{episode[:-2]}.npy")
        else:
            desc_path = os.path.join(self.desc_root, f"{episode[:-1]}.npy")

        desc = np.load(desc_path)
        desc = torch.tensor(desc[:self.desc_max_len], dtype=torch.float32)
        return x, y, subject_id, desc
