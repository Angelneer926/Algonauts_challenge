import torch

def collate_fn_with_desc(batch):
    xs, ys, sids, descs = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    sids = torch.stack(sids)

    max_len = max(d.shape[0] for d in descs)
    padded_descs, desc_masks = [], []

    for d in descs:
        pad_len = max_len - d.shape[0]
        padded = torch.cat([d, torch.zeros(pad_len, d.shape[1])], dim=0)
        mask = torch.cat([torch.zeros(d.shape[0]), torch.ones(pad_len)])
        padded_descs.append(padded)
        desc_masks.append(mask)

    desc_tensor = torch.stack(padded_descs)        # [B, L, D]
    desc_mask = torch.stack(desc_masks).bool()     # [B, L]
    return xs, ys, sids, desc_tensor, desc_mask
