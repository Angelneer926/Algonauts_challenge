import os
import numpy as np
from utils.io import find_stim_path, get_h5_path, load_fmri_for_episode

def load_features(stim_root, fmri_root, subject, episode, modalities):
    stim_features = []
    for modality in modalities:
        modality_root = os.path.join(stim_root, modality)
        stim_path = find_stim_path(modality_root, episode)
        stim_features.append(np.load(stim_path))

    min_len = min(f.shape[0] for f in stim_features)
    stim_features = [f[:min_len] for f in stim_features]
    stim = np.concatenate(stim_features, axis=1)

    h5_path = get_h5_path(fmri_root, subject, episode)
    fmri = load_fmri_for_episode(h5_path, episode)

    return stim, fmri


def load_features_test(stim_root, subject, episode, modalities):
    stim_features = []
    for modality in modalities:
        modality_root = os.path.join(stim_root, modality)
        stim_path = find_stim_path(modality_root, episode)
        stim_features.append(np.load(stim_path))

    min_len = min(f.shape[0] for f in stim_features)
    stim_features = [f[:min_len] for f in stim_features]
    stim = np.concatenate(stim_features, axis=1)
    return stim
