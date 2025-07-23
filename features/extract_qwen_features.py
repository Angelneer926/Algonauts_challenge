import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_qwen_model(model_name="Qwen/Qwen1.5-0.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer


def extract_qwen_layer_features(
    tsv_path,
    model,
    tokenizer,
    num_used_tokens=2048,
    layer_index=12,
    device="cuda",
    mode="current"
):
    df = pd.read_csv(tsv_path, sep="\t")
    text_col = "text_per_tr"
    if text_col not in df.columns:
        print(f"Warning: Column '{text_col}' not found in {tsv_path}, using empty captions.")
        df[text_col] = ""
    df.insert(loc=0, column="is_na", value=df[text_col].isna())

    tokens = []
    outputs_list = []

    for i in tqdm(range(df.shape[0]), desc=f"Processing {Path(tsv_path).stem}"):
        if not df.iloc[i]["is_na"]:
            tr_text = str(df.iloc[i][text_col]) if not pd.isna(df.iloc[i][text_col]) else ""
            if tr_text.strip():
                tokens.append(tr_text)

        context_text = " ".join(tokens[-num_used_tokens:])
        if not context_text.strip():
            pooled = np.zeros(model.config.hidden_size, dtype=np.float32)
        else:
            inputs = tokenizer(
                context_text,
                return_tensors="pt",
                truncation=True,
                max_length=num_used_tokens,
                padding=False
            ).to(device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[layer_index][0]  

                if mode == "current":
                    selected = hidden[-10:] if hidden.shape[0] >= 10 else hidden
                elif mode == "history":
                    selected = hidden[1:-1] if hidden.shape[0] > 2 else hidden
                else:
                    raise ValueError(f"Invalid mode: {mode}. Use 'current' or 'history'.")

                if selected.shape[0] > 0:
                    pooled = selected.mean(dim=0).cpu().numpy()
                else:
                    pooled = np.zeros(model.config.hidden_size, dtype=np.float32)

        outputs_list.append(pooled)

    return np.array(outputs_list, dtype=np.float32)


def save_npy(array, episode_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{episode_name}.npy")
    np.save(out_path, array)
    print(f"Saved: {out_path}")
    print(f"Shape: {array.shape}")


def extract_features_for_season(
    season_dir,
    save_dir,
    num_used_tokens=2048,
    layer_index=12,
    model_name="Qwen/Qwen1.5-0.5B",
    mode="current"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_qwen_model(model_name)

    season_dir = Path(season_dir)
    save_dir = Path(save_dir)

    for tsv_file in sorted(season_dir.glob("*.tsv")):
        episode_name = tsv_file.stem
        out_path = save_dir / f"{episode_name}.npy"
        if out_path.exists():
            print(f"Skipping {episode_name}, already exists.")
            continue

        features = extract_qwen_layer_features(
            tsv_path=str(tsv_file),
            model=model,
            tokenizer=tokenizer,
            num_used_tokens=num_used_tokens,
            layer_index=layer_index,
            device=device,
            mode=mode
        )
        save_npy(features, episode_name, str(save_dir))


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcript_root = os.path.join(root, "stimuli", "transcripts")
    output_root = os.path.join(root, "stimuli", "stimulus_features", "raw") 
    # output_root = os.path.join(root, "stimuli", "stimulus_features", "ood") # ood test dataset
    mode = "current"  # "current" for last-10-token; "history" for all-token
    subfolder = "language_current" if mode == "current" else "language"
    layer_index = 12
    num_used_tokens = 2048
    model_name = "Qwen/Qwen1.5-0.5B"

    # Friends Seasons
    for season_id in range(1, 8):
        season_dir = os.path.join(transcript_root, "friends", f"s{season_id}")
        save_dir = os.path.join(output_root, subfolder, f"season{season_id}")
        extract_features_for_season(
            season_dir=season_dir,
            save_dir=save_dir,
            num_used_tokens=num_used_tokens,
            layer_index=layer_index,
            model_name=model_name,
            mode=mode
        )
    # Movie10
    movie_list = ["bourne", "figures", "life", "wolf"]
    for movie_name in movie_list:
        movie_dir = os.path.join(transcript_root, "movie10", movie_name)
        save_dir = os.path.join(output_root, subfolder, movie_name)
        extract_features_for_season(
            season_dir=movie_dir,
            save_dir=save_dir,
            num_used_tokens=num_used_tokens,
            layer_index=layer_index,
            model_name=model_name,
            mode=mode
        )

    # OOD Movies
    # movie_list = ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]
    # for movie_name in movie_list:
    #     movie_dir = os.path.join(transcript_root, "ood", movie_name)
    #     save_dir = os.path.join(output_root, subfolder, movie_name)
    #     extract_features_for_season(
    #         season_dir=movie_dir,
    #         save_dir=save_dir,
    #         num_used_tokens=num_used_tokens,
    #         layer_index=layer_index,
    #         model_name=model_name,
    #         mode=mode
    #     )
