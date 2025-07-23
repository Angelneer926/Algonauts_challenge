import os
import numpy as np
import re
from glob import glob
from collections import defaultdict

def clean_and_pad_prediction(pred_path, target_len):

    pred = np.load(pred_path)  # (T, D)
    nan_mask = np.isnan(pred)
    if np.any(nan_mask):
        mean_val = np.nanmean(pred, axis=0)
        pred[nan_mask] = np.take(mean_val, np.where(nan_mask)[1])
    else:
        mean_val = np.mean(pred, axis=0)

    cur_len = pred.shape[0]
    if cur_len < target_len:
        pad_len = target_len - cur_len
        pad_block = np.tile(mean_val, (pad_len, 1))
        pred = np.concatenate([pred, pad_block], axis=0)
    elif cur_len > target_len:
        pred = pred[:target_len]
    return pred

def extract_episode_from_filename(filename):

    match = re.search(r"_([a-z0-9]+)_pred\.npy", filename)
    # match = re.search(r"_ood_(.*?)_pred\.npy", filename) # ood test dataset

    return match.group(1) if match else None

def build_submission_dict(pred_root, base_fmri_root):

    submission = defaultdict(dict)

    for pred_path in sorted(glob(os.path.join(pred_root, "*_pred.npy"))):
        fname = os.path.basename(pred_path)
        subject_id = f"sub-{fname[:2]}"  
        episode = extract_episode_from_filename(fname)

        if episode is None:
            print(f"[Warning] Cannot extract episode from: {fname}")
            continue

        sample_len_file = os.path.join(
            base_fmri_root, subject_id, "target_sample_number",
            f"{subject_id}_friends-s7_fmri_samples.npy"
            # f"{subject_id}_ood_fmri_samples.npy" # ood test dataset
        )

        if not os.path.exists(sample_len_file):
            print(f"[Warning] Sample length file not found: {sample_len_file}")
            continue

        sample_lens = np.load(sample_len_file, allow_pickle=True).item()
        if episode not in sample_lens:
            print(f"[Warning] Episode {episode} not found in {sample_len_file}")
            continue

        target_len = sample_lens[episode]
        cleaned_pred = clean_and_pad_prediction(pred_path, target_len)
        submission[subject_id][episode] = cleaned_pred

    return submission

if __name__ == "__main__":
    pred_root = os.path.join("test_predictions")
    base_fmri_root = os.path.join("fmri")

    submission_dict = build_submission_dict(pred_root, base_fmri_root)

    save_path = "friends_submission_dict_1.npy"
    # save_path = "friends_submission_dict_1.npy" # ood test dataset
    np.save(save_path, submission_dict)
    print(f"Submission dict saved to: {save_path}")
