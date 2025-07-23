import os
import json
import h5py
import numpy as np
from tqdm import tqdm
from utils.feature import load_features, load_features_test


def append_to_h5(h5_file, stim_data, fmri_data):
    stim_shape = stim_data.shape
    fmri_shape = fmri_data.shape

    for key, data, shape in zip(["stimuli", "fmri"], [stim_data, fmri_data], [stim_shape, fmri_shape]):
        if key not in h5_file:
            maxshape = (None,) + shape[1:]
            h5_file.create_dataset(key, data=data, maxshape=maxshape, chunks=True, compression="gzip")
        else:
            old_len = h5_file[key].shape[0]
            new_len = old_len + shape[0]
            h5_file[key].resize(new_len, axis=0)
            h5_file[key][old_len:new_len] = data


def build_dataset_h5(subject_list, episode_list, stim_root, fmri_root,
                     h5_save_path, meta_save_path,
                     stim_window=40, fmri_window=35, stride=5, hrf_delay=5,
                     modalities=["language", "audio", "audio_layer3", "language_visual", "language_current", "visual_avg"],
                     log_missing_path=None):
    all_meta = []
    missing = []

    os.makedirs(os.path.dirname(h5_save_path), exist_ok=True)
    h5_file = h5py.File(h5_save_path, "w")

    for subject in subject_list:
        for episode in tqdm(episode_list, desc=f"sub-{subject:02}"):
            try:
                stim, fmri = load_features(stim_root, fmri_root, subject, episode, modalities)
            except (FileNotFoundError, KeyError) as e:
                missing.append((f"sub-{subject:02}", episode, str(e)))
                continue

            max_frame = min(len(stim), len(fmri))
            limit = min(
                max_frame - stim_window,
                max_frame - hrf_delay - fmri_window
            )
            limit = (limit // stride) * stride

            stim_chunks, fmri_chunks = [], []
            for t in range(0, limit + 1, stride):
                stim_chunk = stim[t:t + stim_window]
                fmri_chunk = fmri[t + hrf_delay : t + hrf_delay + fmri_window]
                if stim_chunk.shape[0] == stim_window and fmri_chunk.shape[0] == fmri_window:
                    stim_chunks.append(stim_chunk)
                    fmri_chunks.append(fmri_chunk)
                    all_meta.append({
                        "subject": f"sub-{subject:02}",
                        "episode": episode,
                        "stim_start": t,
                        "fmri_start": t + hrf_delay
                    })

            if stim_chunks:
                stim_chunks = np.stack(stim_chunks)
                fmri_chunks = np.stack(fmri_chunks)
                append_to_h5(h5_file, stim_chunks, fmri_chunks)

    h5_file.close()

    with open(meta_save_path, "w") as f:
        json.dump(all_meta, f)
    print(f"Saved meta: {meta_save_path}")

    if log_missing_path:
        with open(log_missing_path, "w") as f:
            json.dump(missing, f, indent=2)
        print(f"Missing entries saved to: {log_missing_path}")


def build_dataset(subject_list, episode_list, stim_root, fmri_root,
                  stim_window=40, fmri_window=35, stride=5, hrf_delay=5,
                  modalities=["language", "audio", "audio_layer3", "language_visual", "language_current", "visual_avg"],
                  log_missing_path=None):
    all_stim, all_fmri, all_meta = [], [], []
    missing = []

    for subject in subject_list:
        for episode in tqdm(episode_list, desc=f"sub-{subject:02}"):
            try:
                stim, fmri = load_features(stim_root, fmri_root, subject, episode, modalities)
            except (FileNotFoundError, KeyError) as e:
                missing.append((f"sub-{subject:02}", episode, str(e)))
                continue

            max_frame = min(len(stim), len(fmri))
            limit = min(
                max_frame - stim_window,
                max_frame - hrf_delay - fmri_window
            )
            limit = (limit // stride) * stride

            for t in range(0, limit + 1, stride):
                stim_chunk = stim[t:t + stim_window]
                fmri_chunk = fmri[t + hrf_delay : t + hrf_delay + fmri_window]
                if stim_chunk.shape[0] == stim_window and fmri_chunk.shape[0] == fmri_window:
                    all_stim.append(stim_chunk)
                    all_fmri.append(fmri_chunk)
                    all_meta.append({
                        "subject": f"sub-{subject:02}",
                        "episode": episode,
                        "stim_start": t,
                        "fmri_start": t + hrf_delay
                    })

    if log_missing_path:
        with open(log_missing_path, "w") as f:
            json.dump(missing, f, indent=2)
        print(f"Missing entries saved to: {log_missing_path}")

    return {
        "stimuli": np.stack(all_stim),
        "fmri": np.stack(all_fmri),
        "meta": all_meta
    }


def build_dataset_test_h5(subject_list, episode_list, stim_root, fmri_root,
                          h5_save_path, meta_save_path,
                          stim_window=40, fmri_window=35, stride=5, hrf_delay=5,
                          modalities=["language", "audio", "audio_layer3", "language_visual", "language_current", "visual_avg"],
                          log_missing_path=None):
    all_meta = []
    missing = []

    os.makedirs(os.path.dirname(h5_save_path), exist_ok=True)
    h5_file = h5py.File(h5_save_path, "w")

    for subject in subject_list:
        for episode in tqdm(episode_list, desc=f"sub-{subject:02}"):
            try:
                stim = load_features_test(stim_root, subject, episode, modalities)
            except (FileNotFoundError, KeyError) as e:
                missing.append((f"sub-{subject:02}", episode, str(e)))
                continue

            max_frame = len(stim)
            limit = min(
                max_frame - stim_window,
                max_frame - hrf_delay - fmri_window
            )
            limit = (limit // stride) * stride

            stim_chunks, fmri_chunks = [], []
            for t in range(0, limit + 1, stride):
                stim_chunk = stim[t:t + stim_window]
                fmri_chunk = np.zeros((fmri_window, 1000), dtype=np.float32)
                if stim_chunk.shape[0] == stim_window:
                    stim_chunks.append(stim_chunk)
                    fmri_chunks.append(fmri_chunk)
                    all_meta.append({
                        "subject": f"sub-{subject:02}",
                        "episode": episode,
                        "stim_start": t,
                        "fmri_start": t + hrf_delay
                    })

            if stim_chunks:
                stim_chunks = np.stack(stim_chunks)
                fmri_chunks = np.stack(fmri_chunks)
                append_to_h5(h5_file, stim_chunks, fmri_chunks)

    h5_file.close()

    with open(meta_save_path, "w") as f:
        json.dump(all_meta, f)
    print(f"Saved meta: {meta_save_path}")

    if log_missing_path:
        with open(log_missing_path, "w") as f:
            json.dump(missing, f, indent=2)
        print(f"Missing entries saved to: {log_missing_path}")


def collect_existing_episodes(stim_root, modalities):
    episode_sets = []
    for modality in modalities:
        mod_path = os.path.join(stim_root, modality)
        if not os.path.exists(mod_path):
            continue

        episodes = set()
        for root, dirs, files in os.walk(mod_path):
            for f in files:
                if f.endswith(".npy"):
                    episodes.add(f.replace(".npy", ""))
        episode_sets.append(episodes)

    return set.intersection(*episode_sets) if episode_sets else set()


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    stim_root = os.path.join(root, "stimuli", "stimulus_features", "raw")
    # stim_root = os.path.join(root, "stimuli", "stimulus_features", "ood") # ood test dataset
    fmri_root = os.path.join(root, "fmri")
    output_root = root

    subject_list = [1, 2, 3, 5]
    modalities = ["language", "audio", "audio_layer3", "language_visual", "language_current", "visual_avg"]
    movie_list = ["bourne", "figures", "life", "wolf"]
    # movie_list = ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"] # ood test dataset

    all_episodes = collect_existing_episodes(stim_root, modalities)

    train_eps = [f"friends_s0{season}e{ep:02}{part}"
                 for season in range(1, 6)
                 for ep in range(1, 26)
                 for part in ["a", "b", "c", "d"]]
    train_eps += [f"friends_s06e{ep:02}{part}" for ep in range(1, 26) for part in ["a", "c", "d"]]
    train_eps += [f"movie10_{movie}{i:02}" for movie in movie_list for i in range(3, 18)]
    train_eps = [ep for ep in train_eps if ep in all_episodes]

    val_eps = [f"friends_s06e{ep:02}b" for ep in range(1, 26)]
    val_eps += [f"movie10_{movie}{i:02}" for movie in movie_list for i in [1, 2]]
    val_eps = [ep for ep in val_eps if ep in all_episodes]

    test_eps = [f"friends_s07e{ep:02}{part}"
                for ep in range(1, 26)
                for part in ["a", "b", "c", "d"]]
    # test_eps = [f"ood_{movie}{i}" for movie in movie_list for i in [1, 2]] # ood test dataset
    test_eps = [ep for ep in test_eps if ep in all_episodes]


    build_dataset_h5(subject_list, train_eps, stim_root, fmri_root,
                     os.path.join(output_root, "train_data.h5"),
                     os.path.join(output_root, "train_data_meta.json"),
                     modalities=modalities)

    build_dataset_h5(subject_list, val_eps, stim_root, fmri_root,
                     os.path.join(output_root, "val_data.h5"),
                     os.path.join(output_root, "val_data_meta.json"),
                     modalities=modalities)

    build_dataset_test_h5(subject_list, test_eps, stim_root, fmri_root,
                          os.path.join(output_root, "test_data.h5"),
                          os.path.join(output_root, "test_data_meta.json"),
                          modalities=modalities)


if __name__ == "__main__":
    main()
