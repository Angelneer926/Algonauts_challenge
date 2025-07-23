import os
import numpy as np
from pathlib import Path
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
from matplotlib.colors import LinearSegmentedColormap

def plot_mean_encoding_accuracy(corr_dir, subject_list, atlas_subject, atlas_dir, modality="val"):
    all_corrs = []

    for subject in subject_list:
        sid = f"{subject:02d}"
        corr_path = corr_dir / f"sub-{sid}_parcel_corr.npy"
        if not corr_path.exists():
            print(f"Warning: Missing correlation file for sub-{sid}")
            continue

        corrs = np.load(corr_path)
        all_corrs.append(corrs)

    if not all_corrs:
        print("No valid correlation files found.")
        return

    all_corrs = np.stack(all_corrs, axis=0)
    mean_corr = np.nanmean(all_corrs, axis=0)
    mean_r = np.round(np.nanmean(mean_corr), 3)

    atlas_file = f"sub-0{atlas_subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
    atlas_path = atlas_dir / f"sub-0{atlas_subject}" / "atlas" / atlas_file

    masker = NiftiLabelsMasker(labels_img=str(atlas_path))
    masker.fit()
    mean_corr_nii = masker.inverse_transform(mean_corr)

    custom_cmap = LinearSegmentedColormap.from_list(
        "white_yellow_red",
        ["#ffffff", "#ffffff", "#ffffff", "#ffffff", "#ffff00", "#ff0000", "#000000"]
    )
    title = f"Encoding accuracy {modality}, sub-average, mean accuracy: {mean_r:.4f}"
    output_path = corr_dir / f"mean_encoding_accuracy_{modality.replace(' ', '_')}.png"

    display = plotting.plot_glass_brain(
        mean_corr_nii,
        display_mode="lyrz",
        cmap=custom_cmap,
        colorbar=True,
        plot_abs=False,
        symmetric_cbar=False,
        title=title
    )
    display._cbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)

    display.savefig(str(output_path))
    print(f"Saved glass brain to: {output_path}")
    print("Colorbar ticks:", display._cbar.get_ticks())
    plotting.show()

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    corr_dir = root / "val_parcel_corrs_friends_s6b"  # or val_parcel_corrs_movie10
    atlas_dir = root / "fmri"
    subject_list = [1, 2, 3, 5]
    atlas_subject = 1

    plot_mean_encoding_accuracy(
        corr_dir=corr_dir,
        subject_list=subject_list,
        atlas_subject=atlas_subject,
        atlas_dir=atlas_dir,
        modality="Friends s6 (b part)" # or movie 10
    )
