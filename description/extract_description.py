import os
import re
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def extract_sentence_embeddings(sentences, model, tokenizer, device):
    embeddings = []
    for s in tqdm(sentences, desc="Extracting sentence embeddings", leave=False):
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = outputs.pooler_output[0].cpu().numpy()
        embeddings.append(pooled)
    return np.array(embeddings, dtype=np.float32)

def save_embeddings(episode_prefix, sentence_embeddings, save_root):
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, f"{episode_prefix}.npy")
    np.save(save_path, sentence_embeddings)
    print(f"[Saved] {save_path}: shape = {sentence_embeddings.shape}")

def main():
    root = os.path.dirname(os.path.abspath(__file__))  

    description_json_path = os.path.join(root, "description.json")
    # description_json_path = os.path.join(root, "description_ood.json") # ood test dataset

    project_root = os.path.abspath(os.path.join(root, ".."))
    save_root = os.path.join(project_root, "stimuli", "stimulus_features", "raw", "description_full")
    # save_root = os.path.join(project_root, "stimuli", "stimulus_features", "ood", "description_full") # ood test dataset

    model_name = "bert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    with open(description_json_path, "r") as f:
        description_map = json.load(f)

    for episode_prefix, description_text in description_map.items():
        try:
            print(f"Processing {episode_prefix}...")
            sentences = split_sentences(description_text)
            sentence_embeddings = extract_sentence_embeddings(sentences, model, tokenizer, device)
            save_embeddings(episode_prefix, sentence_embeddings, save_root)
        except Exception as e:
            print(f"Failed to process {episode_prefix}: {e}")

if __name__ == "__main__":
    main()
