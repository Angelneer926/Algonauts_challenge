import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.encoder_decoder_transformer_subject import EncoderDecoderTransformer
from dataset.fmri_dataset import FMRIH5Dataset
from utils.dataloader import collate_fn_with_desc
from utils.loss import pearson_corr_loss

def train_model(h5_path, meta_path, model, device, desc_root,
                batch_size=32, num_epochs=10, lr=1e-4, lambda_pearson=0.1):
    dataset = FMRIH5Dataset(h5_path, meta_path, desc_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, collate_fn=collate_fn_with_desc)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
    # for epoch in range(3, 10): # training from checkpoint
        total_mse, total_pearson = 0, 0
        teacher_forcing_ratio = max(0.0, 1.0 - epoch / num_epochs)

        for x, y_out, sids, desc, desc_mask in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x, y_out = x.to(device), y_out.to(device)
            sids = sids.to(device)
            desc, desc_mask = desc.to(device), desc_mask.to(device)

            pred_mse, pred_pearson = model.forward_autoregressive(
                x, y_out, sids, desc=desc, desc_mask=desc_mask,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            loss_mse = nn.functional.mse_loss(pred_mse, y_out)
            loss_pearson = pearson_corr_loss(pred_pearson, y_out)
            loss = loss_mse + lambda_pearson * loss_pearson

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse += loss_mse.item()
            total_pearson += loss_pearson.item()

        print(f"[Epoch {epoch+1}] MSE: {total_mse / len(loader):.4f} | Pearson: {total_pearson / len(loader):.4f}")

        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_checkpoint(os.path.join(ckpt_dir, f"seq2seq_dualattn_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Resume training from checkpoint 
    # ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "seq2seq_dualattn_final_epoch3.pt")
    # if os.path.exists(ckpt_path):
    #     model.load_checkpoint(ckpt_path)
    #     print(f"[Info] Loaded checkpoint from {ckpt_path}")

    train_model(
        h5_path="train_data.h5",
        meta_path="train_data_meta.json",
        model=model,
        device=device,
        desc_root=os.path.join("stimuli", "stimulus_features", "raw", "description_full"),
        batch_size=32,
        num_epochs=10,
        lr=1e-4,
        lambda_pearson=0.15
    )
