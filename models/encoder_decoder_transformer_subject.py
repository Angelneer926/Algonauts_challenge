import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.embedding(pos)
class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.max_len = max_len
        self.rel_embedding = nn.Embedding(2 * max_len - 1, dim)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        rel_pos = pos[None, :] - pos[:, None]  # (T, T)
        rel_pos = rel_pos.clamp(-self.max_len + 1, self.max_len - 1)
        rel_pos += self.max_len - 1

        rel_emb = self.rel_embedding(rel_pos)  # (T, T, D)
        rel_sum = rel_emb.sum(dim=1)  # (T, D)
        rel_sum = rel_sum.unsqueeze(0).expand(B, -1, -1)  # (B, T, D)
        return x + rel_sum

class TransformerDecoderLayerWithDualCrossAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_stim = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_desc = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, tgt, memory_stim, memory_desc, tgt_mask=None, desc_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn_stim(tgt, memory_stim, memory_stim)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn_desc(tgt, memory_desc, memory_desc, key_padding_mask=desc_mask)[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, num_parcels, num_subjects=4, max_len=1024,
                 use_learnable_bos=True, predict_residual=False, activation="relu", dropout=0.1,
                 use_desc_memory=True, desc_dim=768, max_desc_len=32):
        super().__init__()
        self.use_learnable_bos = use_learnable_bos
        self.predict_residual = predict_residual
        self.num_parcels = num_parcels
        self.use_desc_memory = use_desc_memory

        self.encoder_input_proj = nn.Linear(input_dim, hidden_dim)
        self.decoder_input_proj = nn.Linear(num_parcels, hidden_dim)

        self.encoder_pos_enc = RelativePositionalEncoding(hidden_dim, max_len)
        self.decoder_pos_enc = RelativePositionalEncoding(hidden_dim, max_len)

        if use_desc_memory:
            self.desc_proj = nn.Linear(desc_dim, hidden_dim)
            self.desc_pos_enc = RelativePositionalEncoding(hidden_dim, max_desc_len)

        if use_learnable_bos:
            self.learnable_bos = nn.Parameter(torch.zeros(1, 1, num_parcels))
        else:
            self.register_buffer("bos_token", torch.zeros(1, 1, num_parcels))

        self.subject_embeddings = nn.Embedding(num_subjects, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim*4, dropout=dropout, activation=activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerWithDualCrossAttn(hidden_dim, nhead, hidden_dim*4, dropout, activation)
            for _ in range(num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.subject_output_layers_mse = nn.ModuleList([nn.Linear(hidden_dim, num_parcels) for _ in range(num_subjects)])
        self.subject_output_layers_pearson = nn.ModuleList([nn.Linear(hidden_dim, num_parcels) for _ in range(num_subjects)])

    def forward_autoregressive(self, src, tgt, subject_ids, desc, desc_mask=None, teacher_forcing_ratio=1.0):
        if src.ndim == 4:
            B, T, W, D = src.shape
            src = src.view(B, T, W * D)

        B, T_tgt, _ = tgt.shape
        device = tgt.device

        src_proj = self.encoder_input_proj(src)
        src_proj = self.encoder_pos_enc(src_proj)
        memory = self.encoder(src_proj)

        desc_proj = self.desc_proj(desc)
        desc_proj = self.desc_pos_enc(desc_proj)

        outputs_mse = []
        outputs_pearson = []
        projected_inputs = []

        if self.use_learnable_bos:
            input_token = self.learnable_bos.expand(B, -1, -1)
        else:
            input_token = self.bos_token.expand(B, -1, -1)

        input_token = self.decoder_input_proj(input_token)
        subject_embed = self.subject_embeddings(subject_ids).unsqueeze(1)
        input_token = input_token + subject_embed
        projected_inputs.append(input_token)

        for t in range(T_tgt):
            tgt_seq = torch.cat(projected_inputs, dim=1)
            tgt_proj = self.decoder_pos_enc(tgt_seq)

            for layer in self.decoder_layers:
                tgt_proj = layer(tgt_proj, memory, desc_proj, desc_mask=desc_mask)

            dec_output = tgt_proj[:, -1]

            mse_pred = torch.stack([
                self.subject_output_layers_mse[subject_ids[i]](dec_output[i]) for i in range(B)
            ], dim=0).unsqueeze(1)

            pearson_pred = torch.stack([
                self.subject_output_layers_pearson[subject_ids[i]](dec_output[i]) for i in range(B)
            ], dim=0).unsqueeze(1)

            outputs_mse.append(mse_pred)
            outputs_pearson.append(pearson_pred)

            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                next_token = self.decoder_input_proj(tgt[:, t:t+1, :])
            else:
                next_token = self.decoder_input_proj(mse_pred.detach())

            next_token = next_token + subject_embed
            projected_inputs.append(next_token)

        return torch.cat(outputs_mse, dim=1), torch.cat(outputs_pearson, dim=1)

    def generate(self, src, tgt_len, subject_ids, desc, desc_mask=None):
        if src.ndim == 4:
            B, T, W, D = src.shape
            src = src.view(B, T, W * D)
        else:
            B = src.size(0)

        src_proj = self.encoder_input_proj(src)
        src_proj = self.encoder_pos_enc(src_proj)
        memory = self.encoder(src_proj)

        desc_proj = self.desc_proj(desc)
        desc_proj = self.desc_pos_enc(desc_proj)

        input_token = self.learnable_bos.expand(B, -1, -1).to(src.device) if self.use_learnable_bos else self.bos_token.expand(B, -1, -1).to(src.device)
        input_token = self.decoder_input_proj(input_token)
        subject_embed = self.subject_embeddings(subject_ids).unsqueeze(1)
        input_token = input_token + subject_embed

        projected_inputs = [input_token]
        outputs = []

        for _ in range(tgt_len):
            tgt_seq = torch.cat(projected_inputs, dim=1)
            tgt_proj = self.decoder_pos_enc(tgt_seq)

            for layer in self.decoder_layers:
                tgt_proj = layer(tgt_proj, memory, desc_proj, desc_mask=desc_mask)

            dec_output = tgt_proj[:, -1]
            pred = torch.stack([
                self.subject_output_layers_mse[subject_ids[i]](dec_output[i])
                for i in range(B)
            ], dim=0).unsqueeze(1)

            outputs.append(pred)
            input_token = self.decoder_input_proj(pred)
            input_token = input_token + subject_embed
            projected_inputs.append(input_token)

        return torch.cat(outputs, dim=1)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))