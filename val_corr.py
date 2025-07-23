import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from models.encoder_decoder_transformer_subject import EncoderDecoderTransformer
from dataset.val_dataset import ValDataset
from utils.eval import (
    evaluate_prediction_correlation_concat,
    evaluate_movie10_prediction_correlation,
)

def generate_val_predictions(model_path, h5_path, meta_path, desc_root,
                             output_dir="val_predictions", device="cuda", tgt_len=35):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = EncoderDecoderTransformer(
        input_dim=6656,
        hidden_dim=512,
        num_layers=2,
        nhead=4,
        num_parcels=1000,
        num_subjects=4,
        max_len=1024,
        use_learnable_bos=True,
        predict_residual=False,
        use_desc_memory=True,
        desc_dim=768,
        max_desc_len=64
    )
    model.load_checkpoint(model_path)
    model.to(device)
    model.eval()

    dataset = ValDataset(h5_path, meta_path, desc_root, max_desc_len=64)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    recon = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for x, subject_id, subject_str, ep, pos, desc, mask in tqdm(loader, desc="Generating"):
            x = x.to(device)
            subject_id = subject_id.to(device)
            desc = desc.to(device)
            mask = mask.to(device)

            pred = model.generate(
                x, tgt_len=tgt_len, subject_ids=subject_id,
                desc=desc, desc_mask=mask
            )[0].cpu().numpy()

            subject = subject_str[0].replace("sub-", "")
            episode = ep[0]
            start_pos = pos[0].item()
            key = f"{subject}_{episode}"

            for i in range(tgt_len):
                frame_idx = start_pos + i
                recon[key][frame_idx].append(pred[i])

    os.makedirs(output_dir, exist_ok=True)
    for key, frame_dict in recon.items():
        max_len = max(frame_dict.keys()) + 1
        output = np.full((max_len, 1000), np.nan)
        for i, frames in frame_dict.items():
            output[i] = np.mean(frames, axis=0)
        np.save(os.path.join(output_dir, f"{key}_pred.npy"), output)

    print(f"Saved all predictions to {output_dir}")


if __name__ == "__main__":
    generate_val_predictions(
        model_path="checkpoints/seq2seq_dualattn_epoch8.pt", #ckpt can be changed
        h5_path="val_data.h5",
        meta_path="val_data_meta.json",
        desc_root=os.path.join("stimuli", "stimulus_features", "raw", "description_full"),
        output_dir="val_predictions",
        device="cuda",
        tgt_len=35
    )

    evaluate_prediction_correlation_concat(
        prediction_dir="val_predictions",
        fmri_dir="/net/projects/ycleong/heqianyi926/Algonauts_Challenge/fmri",
        subjects=[1, 2, 3, 5],
        trim_end=5
    )

    evaluate_movie10_prediction_correlation(
        prediction_dir="val_predictions",
        fmri_dir="/net/projects/ycleong/heqianyi926/Algonauts_Challenge/fmri",
        subjects=[1, 2, 3, 5],
        trim_end=5
    )

    # test
    # generate_val_predictions(
    #     model_path="checkpoints/seq2seq_dualattn_epoch8.pt", #ckpt can be changed
    #     h5_path="test_data.h5",
    #     meta_path="test_data_meta.json",
    #     desc_root=os.path.join("stimuli", "stimulus_features", "ood", "description_full"), # friend s7 replace ood to raw
    #     output_dir="test_predictions",
    #     device="cuda",
    #     tgt_len=35
    # )
