import os
import json
import h5py
import glob
import numpy as np

def find_stim_path(modality_root, episode):
    pattern = os.path.join(modality_root, "**", episode + ".npy")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"Stimulus not found: {pattern}")
    return matches[0]


def load_fmri_for_episode(h5_path, episode):
    if "friends" in episode:
        target_suffix = "task-" + episode.replace("friends_", "")
    elif "movie10" in episode:
        target_suffix = "task-" + episode.replace("movie10_", "")
    else:
        raise ValueError(f"Unrecognized episode format: {episode}")

    with h5py.File(h5_path, "r") as f:
        if "movie10" in episode:
            run1_keys = [k for k in f.keys() if target_suffix in k and "_run-1" in k]
            if run1_keys:
                return f[run1_keys[0]][:].astype(np.float32)
            no_run_keys = [k for k in f.keys() if target_suffix in k and "_run" not in k]
            if no_run_keys:
                return f[no_run_keys[0]][:].astype(np.float32)
            raise KeyError(f"Could not find suitable key for '{target_suffix}' in {h5_path}")
        else:
            matching_keys = [k for k in f.keys() if target_suffix in k]
            if not matching_keys:
                raise KeyError(f"Could not find key matching '{target_suffix}' in {h5_path}")
            return f[matching_keys[0]][:].astype(np.float32)


def get_h5_path(fmri_root, subject, episode):
    h5_dir = os.path.join(fmri_root, f"sub-{subject:02}", "func")
    all_h5s = glob.glob(os.path.join(h5_dir, "*.h5"))
    for h5_path in all_h5s:
        try:
            _ = load_fmri_for_episode(h5_path, episode)
            return h5_path
        except KeyError:
            continue
    raise FileNotFoundError(f"No HDF5 file contains episode {episode} for subject {subject}")


def save_dataset(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, stimuli=data["stimuli"], fmri=data["fmri"])
    with open(save_path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(data["meta"], f)
    print(f"Saved: {save_path}")

def load_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta
