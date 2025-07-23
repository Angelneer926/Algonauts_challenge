import os
import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import VideoMAEModel, VideoMAEImageProcessor
from pathlib import Path


def extract_videomae_cls_features(video_path, tr, model, processor, device, layer_index=-1, num_frames=16):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps

    start_times = np.arange(0, duration - tr + 1e-2, tr)
    features = []

    for start in tqdm(start_times, desc=f"Extracting {video_path.name}"):
        end = start + tr
        timestamps = np.linspace(start, end, num_frames, endpoint=False)
        frame_indices = (timestamps * fps).astype(int)
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)

        try:
            frames = vr.get_batch(frame_indices).asnumpy()
        except Exception as e:
            print(f"[Skip] {video_path.name} | Frame error: {e}")
            continue

        frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
        inputs = processor(list(frames_tensor), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_index]  # shape: (1, num_tokens, dim)
            cls_feat = hidden[:, 0]  # CLS token
            features.append(cls_feat.squeeze().cpu().numpy())

    return np.array(features, dtype=np.float32)


def apply_cumulative_average_and_concat(features):
    cumsum = np.cumsum(features, axis=0)
    count = np.arange(1, features.shape[0] + 1).reshape(-1, 1)
    avg = cumsum / count
    return np.concatenate([features, avg], axis=1)  # shape (T, 2D)


def batch_process_video_folder(video_dir, save_dir, prefix=None, drop_prefix=False,
                                model_name="MCG-NJU/videomae-large", layer_index=-1, tr=1.49, num_frames=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEModel.from_pretrained(model_name).to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    video_files = sorted([f for f in video_dir.glob("*.mkv")])

    for video_path in video_files:
        base = video_path.stem
        if drop_prefix and prefix and base.startswith(prefix + "_"):
            base = base[len(prefix) + 1:]

        output_name = f"{prefix + '_' if prefix else ''}{base}.npy"
        output_path = save_dir / output_name
        if output_path.exists():
            print(f"Skip {output_path}")
            continue

        try:
            features = extract_videomae_cls_features(
                video_path, tr, model, processor, device, layer_index, num_frames
            )
            final_feat = apply_cumulative_average_and_concat(features)
            np.save(output_path, final_feat)
            print(f"Saved {output_path}")
            print(f"shape: {final_feat.shape}") 
        except Exception as e:
            print(f"Error: {video_path.name}: {e}")


def main():
    root = Path(__file__).resolve().parent.parent
    video_root = root / "stimuli" / "movies"
    save_root_raw = root / "stimuli" / "stimulus_features" / "raw" / "visual_avg"
    save_root_ood = root / "stimuli" / "stimulus_features" / "ood" / "visual_avg"

    # Friends
    for season_id in range(1, 8):
        video_dir = video_root / "friends" / f"s{season_id}"
        save_dir = save_root_raw / f"season{season_id}"
        if video_dir.exists():
            batch_process_video_folder(
                video_dir, save_dir, prefix="friends", drop_prefix=True
            )

    # Movie10
    movie10_list = ["bourne", "figures", "life", "wolf"]
    for movie_name in movie10_list:
        video_dir = video_root / "movie10" / movie_name
        save_dir = save_root_raw / movie_name
        if video_dir.exists():
            batch_process_video_folder(
                video_dir, save_dir, prefix="movie10", drop_prefix=False
            )

    # OOD Movies
    ood_list = ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]
    for movie_name in ood_list:
        video_dir = video_root / "ood" / movie_name
        save_dir = save_root_ood / movie_name
        if video_dir.exists():
            batch_process_video_folder(
                video_dir, save_dir, prefix="ood", drop_prefix=False
            )


if __name__ == "__main__":
    main()
