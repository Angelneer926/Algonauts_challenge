import os
import numpy as np
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torchaudio
from pathlib import Path

def extract_audio_from_chunk(clip, start, duration, tmp_path, sr):
    os.makedirs(tmp_path, exist_ok=True)
    chunk_path = os.path.join(tmp_path, "temp_chunk.wav")
    chunk_clip = clip.subclip(start, start + duration)
    chunk_clip.audio.write_audiofile(chunk_path, verbose=False, logger=None, fps=sr)

    waveform, sample_rate = torchaudio.load(chunk_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform.unsqueeze(0)
    return waveform.squeeze(0), sample_rate

def extract_hubert_features_from_video(video_path, tr, model, feature_extractor,
                                       device, layer_index=9, tmp_path="./tmp_audio", sr=16000):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    n_chunks = int(duration // tr)
    start_times = [i * tr for i in range(n_chunks)]
    features = []

    for start in tqdm(start_times, desc=f"Extracting {os.path.basename(video_path)}"):
        audio_array, sample_rate = extract_audio_from_chunk(clip, start, tr, tmp_path, sr)
        inputs = feature_extractor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_index]
            pooled = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()

        features.append(pooled)

    return np.array(features, dtype=np.float32)

def batch_process_video_folder(video_dir, save_dir, prefix="", layer_index=9, tr=1.49, sr=16000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith((".mkv", ".mp4", ".webm"))])

    for video_file in video_files:
        base = video_file.rsplit(".", 1)[0]
        video_path = os.path.join(video_dir, video_file)
        output_name = f"{prefix}{base}.npy"
        output_path = os.path.join(save_dir, output_name)

        if os.path.exists(output_path):
            print(f"Skip {output_path}")
            continue

        features = extract_hubert_features_from_video(
            video_path, tr, model, feature_extractor, device,
            layer_index=layer_index, tmp_path="./tmp_audio", sr=sr
        )
        np.save(output_path, features)
        print(f"Saved {output_path}")

def main(layer_index=9):
    root = Path(__file__).resolve().parent.parent
    video_base = root / "stimuli" / "movies"

    if layer_index == 3:
        save_name = "audio_layer3"
    elif layer_index == 9:
        save_name = "audio"
    else:
        raise ValueError("layer_index must be either 3 or 9.")

    save_root_raw = root / "stimuli" / "stimulus_features" / "raw" / save_name
    save_root_ood = root / "stimuli" / "stimulus_features" / "ood" / save_name

    for season_id in range(1, 8):
        video_dir = video_base / "friends" / f"s{season_id}"
        save_dir = save_root_raw / f"season{season_id}"
        if video_dir.exists():
            batch_process_video_folder(video_dir, save_dir, prefix="", layer_index=layer_index)

    # Movie10
    movie10_list = ["bourne", "figures", "life", "wolf"]
    for movie_name in movie10_list:
        video_dir = video_base / "movie10" / movie_name
        save_dir = save_root_raw / movie_name
        if video_dir.exists():
            batch_process_video_folder(video_dir, save_dir, prefix="movie10_", layer_index=layer_index)

    # OOD Movies
    ood_list = ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]
    for movie_name in ood_list:
        video_dir = video_base / "ood" / movie_name
        save_dir = save_root_ood / movie_name
        if video_dir.exists():
            batch_process_video_folder(video_dir, save_dir, prefix="ood_", layer_index=layer_index)

if __name__ == "__main__":
    main(layer_index=9)
    #main(layer_index=3) # change to layer3
