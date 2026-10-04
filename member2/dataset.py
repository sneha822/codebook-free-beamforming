# Member 2 - DL Engineer
# Week 2 task: load Member 1's channel data and feed it to the model in batches.
#
# Two things we have to do before the network can use the data:
#   1. split each complex channel into real + imaginary parts
#      (neural networks cannot take complex numbers directly).
#   2. normalise everything, because raw channel values are tiny and raw
#      positions are large - both make learning hard.

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Member 1 saves the data in their folder. We load it from there.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "member1")


def load_raw():
    pos = np.load(os.path.join(DATA_DIR, "positions.npy"))
    h = np.load(os.path.join(DATA_DIR, "channels.npy"))
    return pos, h


def build_loaders(batch_size=512):
    pos, h = load_raw()

    h_re = torch.tensor(h.real, dtype=torch.float32)
    h_im = torch.tensor(h.imag, dtype=torch.float32)
    pos_t = torch.tensor(pos, dtype=torch.float32)

    # --- normalisation (remember these constants, we need them again later) ---
    # scale channels so their average power is about 1
    chan_scale = (h_re ** 2 + h_im ** 2).mean().sqrt()
    h_re = h_re / chan_scale
    h_im = h_im / chan_scale

    # scale positions into [-1, 1]
    pos_min = pos_t.min(0).values
    pos_max = pos_t.max(0).values
    pos_t = 2 * (pos_t - pos_min) / (pos_max - pos_min) - 1

    norm = {"chan_scale": chan_scale, "pos_min": pos_min, "pos_max": pos_max}

    # --- spatial split (block split, not random) ---
    # nearby users have almost identical channels, so a random split would
    # leak test answers into training. Blocks keep nearby users together.
    n = len(pos_t)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    idx = torch.arange(n)
    parts = {
        "train": idx[:n_tr],
        "val":   idx[n_tr:n_tr + n_va],
        "test":  idx[n_tr + n_va:],
    }

    loaders = {}
    for name, ids in parts.items():
        ds = TensorDataset(pos_t[ids], h_re[ids], h_im[ids])
        loaders[name] = DataLoader(ds, batch_size=batch_size,
                                   shuffle=(name == "train"), num_workers=0)
    return loaders, norm


if __name__ == "__main__":
    loaders, norm = build_loaders()
    print("channel scale:", float(norm["chan_scale"]))
    pos_b, hre_b, him_b = next(iter(loaders["train"]))
    print("batch positions:", tuple(pos_b.shape))     # (512, 3)
    print("batch h_re:", tuple(hre_b.shape))           # (512, M)
    print("mean power of batch (~1 is good):",
          float((hre_b ** 2 + him_b ** 2).mean()))
