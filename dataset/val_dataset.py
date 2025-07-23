import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

def load_meta(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)

class ValDataset(Dataset):
    def __init__(self, h5_path, meta_path, desc_root, max_desc_len=32):
        self.desc_root = desc_root
        self.max_desc_len = max_desc_len
        self.h5 = h5py.File(h5_path, "r")
        self.stimuli = self.h5["stimuli"]
        self.meta = load_meta(meta_path)

        self.subject_map = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}
        self.subject_labels = [self.subject_map[m["subject"]] for m in self.meta]

    def __getitem__(self, idx):
        x = torch.tensor(self.stimuli[idx], dtype=torch.float32)
        subject_str = self.meta[idx]["subject"]
        subject_id = torch.tensor(self.subject_labels[idx], dtype=torch.long)
        ep = self.meta[idx]["episode"]
        pos = self.meta[idx]["fmri_start"]

        if "movie10" in ep:
            desc_path = os.path.join(self.desc_root, f"{ep[:-2]}.npy")
        else:
            desc_path = os.path.join(self.desc_root, f"{ep[:-1]}.npy")

        desc = np.load(desc_path)
        desc = torch.tensor(desc[:self.max_desc_len], dtype=torch.float32)

        mask = torch.zeros(self.max_desc_len, dtype=torch.bool)
        if desc.shape[0] < self.max_desc_len:
            pad = torch.zeros(self.max_desc_len - desc.shape[0], desc.shape[1])
            desc = torch.cat([desc, pad], dim=0)
            mask[desc.shape[0]:] = True

        return x, subject_id, subject_str, ep, pos, desc, mask

    def __len__(self):
        return len(self.stimuli)
