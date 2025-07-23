import os
import re
import glob
import h5py
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict

def evaluate_prediction_correlation_concat(prediction_dir, fmri_dir, subjects, trim_end=5):
    subject_corrs = []
    subject_parcel_corrs = {}

    for subject in subjects:
        subject_id = f"{subject:02d}"
        print(f"Evaluating subject {subject_id} ...")

        pattern = os.path.join(
            fmri_dir, f"sub-{subject_id}", "func",
            f"sub-{subject_id}_task-friends*_parcel-1000Par7Net*_bold.h5"
        )

        matches = glob.glob(pattern)
        if not matches:
            print(f"No fMRI file found for subject {subject_id}")
            continue

        fmri_path = matches[0]
        all_gt, all_pred = [], []

        with h5py.File(fmri_path, "r") as f:
            used_movie_keys = set()

            for key in f.keys():
                match_friends = re.search(r"task-(s\d+e\d+[ab])", key)
                match_movie = re.search(r"task-([a-z]+[0-9]{2,})(?:_run-(\d))?", key)

                if match_friends:
                    ep = f"friends_{match_friends.group(1)}"
                elif match_movie:
                    movie_ep = match_movie.group(1)
                    run = match_movie.group(2)
                    ep = f"movie10_{movie_ep}"
                    if run == "2": continue
                    if run == "1": used_movie_keys.add(ep)
                    elif ep in used_movie_keys: continue
                else:
                    continue

                pred_path = os.path.join(prediction_dir, f"{subject_id}_{ep}_pred.npy")
                if not os.path.exists(pred_path):
                    continue

                gt = f[key][:]  
                pred = np.load(pred_path)

                gt = gt[:-trim_end]
                pred = pred[:-trim_end]

                min_len = min(len(gt), len(pred))
                if min_len == 0 or gt.shape[1] != pred.shape[1]:
                    continue

                gt = gt[:min_len]
                pred = pred[:min_len]

                valid_mask = ~np.isnan(pred).any(axis=1)
                gt = gt[valid_mask]
                pred = pred[valid_mask]

                if len(gt) > 0:
                    all_gt.append(gt)
                    all_pred.append(pred)

        if len(all_gt) == 0:
            print(f"No valid episodes for subject {subject_id}")
            continue

        fmri_val = np.concatenate(all_gt, axis=0)
        fmri_val_pred = np.concatenate(all_pred, axis=0)

        parcel_corrs = np.full(1000, np.nan)
        for p in range(fmri_val.shape[1]):
            if np.isnan(fmri_val_pred[:, p]).any():
                continue
            try:
                r, _ = pearsonr(fmri_val[:, p], fmri_val_pred[:, p])
                parcel_corrs[p] = r
            except:
                continue

        mean_r = np.nanmean(parcel_corrs)
        print(f"Subject {subject_id} mean correlation = {mean_r:.4f}")
        subject_corrs.append(mean_r)
        subject_parcel_corrs[subject_id] = parcel_corrs

    if subject_corrs:
        overall = np.mean(subject_corrs)
        print(f"\nOverall mean correlation across subjects = {overall:.4f}")

        os.makedirs("val_parcel_corrs_friends_s6b", exist_ok=True)
        for sid, corrs in subject_parcel_corrs.items():
            np.save(f"val_parcel_corrs_subjectconcat/sub-{sid}_parcel_corr.npy", corrs)
        print(f"\nSaved per-subject per-parcel correlations to val_parcel_corrs_subjectconcat/")
    else:
        print("No valid subjects for evaluation.")


def evaluate_movie10_prediction_correlation(prediction_dir, fmri_dir, subjects, trim_end=5):
    subject_corrs = []
    subject_parcel_corrs = {}

    for subject in subjects:
        subject_id = f"{subject:02d}"
        pattern = os.path.join(
            fmri_dir, f"sub-{subject_id}", "func",
            f"sub-{subject_id}_task-movie10*_parcel-1000Par7Net*_bold.h5"
        )

        matches = glob.glob(pattern)
        if not matches:
            print(f"No fMRI file found for subject {subject_id}")
            continue

        fmri_path = matches[0]
        all_gt, all_pred = [], []
        seen_episodes = set()

        with h5py.File(fmri_path, "r") as f:
            for key in f.keys():
                task_match = re.search(r"task-([a-z]+[0-9]{2,})", key)
                run_match = re.search(r"run-(\d)", key)

                if not task_match:
                    continue

                movie_ep = task_match.group(1)
                run = run_match.group(1) if run_match else None
                ep = f"movie10_{movie_ep}"

                if ep in seen_episodes or run == "2":
                    continue

                pred_path = os.path.join(prediction_dir, f"{subject_id}_{ep}_pred.npy")
                if not os.path.exists(pred_path):
                    continue

                gt = f[key][:]
                pred = np.load(pred_path)

                gt = gt[:-trim_end]
                pred = pred[:-trim_end]

                min_len = min(len(gt), len(pred))
                if min_len == 0 or gt.shape[1] != pred.shape[1]:
                    continue

                gt = gt[:min_len]
                pred = pred[:min_len]

                valid_mask = ~np.isnan(pred).any(axis=1)
                gt = gt[valid_mask]
                pred = pred[valid_mask]

                if len(gt) > 0:
                    all_gt.append(gt)
                    all_pred.append(pred)
                    seen_episodes.add(ep)

        if len(all_gt) == 0:
            print(f"No valid movie10 episodes for subject {subject_id}")
            continue

        fmri_val = np.concatenate(all_gt, axis=0)
        fmri_val_pred = np.concatenate(all_pred, axis=0)

        parcel_corrs = np.full(1000, np.nan)
        for p in range(fmri_val.shape[1]):
            if np.isnan(fmri_val_pred[:, p]).any():
                continue
            try:
                r, _ = pearsonr(fmri_val[:, p], fmri_val_pred[:, p])
                parcel_corrs[p] = r
            except:
                continue

        mean_r = np.nanmean(parcel_corrs)
        print(f"Subject {subject_id} mean movie10 correlation = {mean_r:.4f}")
        subject_corrs.append(mean_r)
        subject_parcel_corrs[subject_id] = parcel_corrs

    if subject_corrs:
        overall = np.mean(subject_corrs)
        print(f"\nOverall mean movie10 correlation across subjects = {overall:.4f}")

        os.makedirs("val_parcel_corrs_movie10", exist_ok=True)
        for sid, corrs in subject_parcel_corrs.items():
            np.save(f"val_parcel_corrs_movie10/sub-{sid}_parcel_corr.npy", corrs)
        print(f"\nSaved per-subject movie10 parcel correlations to val_parcel_corrs_movie10/")
    else:
        print("No valid movie10 subjects for evaluation.")
