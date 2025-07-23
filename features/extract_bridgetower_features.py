import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import BridgeTowerModel, BridgeTowerProcessor
from pathlib import Path

def extract_bridgetower_pooled_features(video_path, transcript_path, tr, model, processor, device, frame_per_chunk=1):
    df = pd.read_csv(transcript_path, sep='\t')
    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    start_times = [x for x in np.arange(0, duration, tr)][:-1]
    features = []

    for i, start in tqdm(enumerate(start_times), total=len(start_times), desc=f"Extracting {video_path.name}"):
        chunk_clip = clip.subclip(start, start + tr)
        frame_times = np.linspace(0, chunk_clip.duration, frame_per_chunk + 2)[1:-1]
        frames = [Image.fromarray(chunk_clip.get_frame(t)) for t in frame_times]

        caption = ""
        if i < len(df) and pd.notna(df.iloc[i]["text_per_tr"]):
            caption = df.iloc[i]["text_per_tr"]

        chunk_feats = []
        for frame in frames:
            inputs = processor(images=frame, text=caption, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_feat = outputs.pooler_output.squeeze(0).cpu().numpy()
                chunk_feats.append(cls_feat)

        pooled_feat = np.mean(chunk_feats, axis=0)
        features.append(pooled_feat)

    return np.array(features, dtype=np.float32)

def batch_process_video_folder(video_dir, transcript_dir, save_dir, prefix="", tr=1.49, frame_per_chunk=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
    model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base").to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    video_files = sorted([f for f in video_dir.glob("*.mkv")])

    for video_path in video_files:
        base = video_path.stem
        transcript_path = transcript_dir / f"{prefix}{base}.tsv"
        output_path = save_dir / f"{prefix}{base}.npy"

        if not transcript_path.exists():
            print(f"Skipping {video_path.name} (no transcript).")
            continue

        if output_path.exists():
            print(f"Skip {output_path}")
            continue

        features = extract_bridgetower_pooled_features(
            video_path, transcript_path, tr,
            model, processor, device, frame_per_chunk
        )
        np.save(output_path, features)
        print(f"Saved {output_path}")

def main():
    root = Path(__file__).resolve().parent.parent
    video_root = root / "stimuli" / "movies"
    transcript_root = root / "stimuli" / "transcripts"
    save_root_raw = root / "stimuli" / "stimulus_features" / "raw" / "language_visual"
    save_root_ood = root / "stimuli" / "stimulus_features" / "ood" / "language_visual"

    # Friends 
    for season_id in range(1, 8):
        video_dir = video_root / "friends" / f"s{season_id}"
        transcript_dir = transcript_root / "friends" / f"s{season_id}"
        save_dir = save_root_raw / f"season{season_id}"
        if video_dir.exists() and transcript_dir.exists():
            batch_process_video_folder(video_dir, transcript_dir, save_dir, prefix="")

    # Movie10 
    movie10_list = ["bourne", "figures", "life", "wolf"]
    for movie_name in movie10_list:
        video_dir = video_root / "movie10" / movie_name
        transcript_dir = transcript_root / "movie10" / movie_name
        save_dir = save_root_raw / movie_name  
        if video_dir.exists() and transcript_dir.exists():
            batch_process_video_folder(video_dir, transcript_dir, save_dir, prefix="movie10_")

    # OOD Movies 
    ood_list = ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]
    for movie_name in ood_list:
        video_dir = video_root / "ood" / movie_name
        transcript_dir = transcript_root / "ood" / movie_name
        save_dir = save_root_ood / movie_name
        if video_dir.exists() and transcript_dir.exists():
            batch_process_video_folder(video_dir, transcript_dir, save_dir, prefix="ood_")

if __name__ == "__main__":
    main()
