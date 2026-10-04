# Member 2 - DL Engineer
# Visual results: trains the model for a bit and draws three figures so we can
# SEE what is happening (numbers alone are hard to present).
#   1. training_curve.png  - our data rate climbing toward the best possible.
#   2. beam_pattern.png    - the AI's beam shape vs the ideal beam (polar plot).
#   3. comparison_bar.png  - AI vs the two baselines (MRT ceiling, DFT floor).
#
# Run from inside the member2 folder:  python visualize_results.py

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # save to file instead of opening a window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "member1"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "member3"))
from model import BeamNet                                   # same folder
from generate_channels import make_dataset                 # Member 1
from baselines import phase_only_upper_bound, dft_codebook_gain, capacity  # Member 1
from loss import neg_rate_loss, array_gain, rate_from_gain, optimal_gain   # Member 3

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
SNR_DB = 10.0
M = 64

# ---- data (Member 1) ----
pos, h = make_dataset(n_users=20000, M=M, seed=0)
chan_scale = np.sqrt((np.abs(h) ** 2).mean())
h_n = h / chan_scale
pos_min, pos_max = pos.min(0), pos.max(0)
pos_n = (2 * (pos - pos_min) / (pos_max - pos_min) - 1).astype(np.float32)

# simple split
n = len(pos); n_tr = int(0.8 * n)
tr = slice(0, n_tr); va = slice(n_tr, n)

pos_t = torch.tensor(pos_n, device=device)
h_re = torch.tensor(h_n.real.astype(np.float32), device=device)
h_im = torch.tensor(h_n.imag.astype(np.float32), device=device)

# ---- train and record the learning curve ----
model = BeamNet(in_dim=3, M=M).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# baselines on the validation set (these are flat reference lines)
h_va = h_n[va]
ceiling = capacity(phase_only_upper_bound(h_va), SNR_DB).mean()
floor = capacity(dft_codebook_gain(h_va), SNR_DB).mean()

epochs = 40
rates = []
for ep in range(epochs):
    model.train()
    # mini-batch over the training split
    perm = torch.randperm(n_tr, device=device)
    for i in range(0, n_tr, 2048):
        b = perm[i:i + 2048]
        opt.zero_grad()
        loss = neg_rate_loss(h_re[b], h_im[b], model(pos_t[b]), SNR_DB)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        g = array_gain(h_re[va], h_im[va], model(pos_t[va]))
        rates.append(rate_from_gain(g, SNR_DB).mean().item())
print(f"final AI rate: {rates[-1]:.3f}  (ceiling {ceiling:.3f}, floor {floor:.3f})")

# ---- figure 1: learning curve ----
plt.figure(figsize=(7, 4.5))
plt.plot(range(1, epochs + 1), rates, marker="o", ms=3, label="AI model")
plt.axhline(ceiling, ls="--", color="green", label="best possible (ceiling)")
plt.axhline(floor, ls="--", color="red", label="DFT codebook (floor to beat)")
plt.xlabel("epoch"); plt.ylabel("spectral efficiency (bits/s/Hz)")
plt.title("The AI learning to aim the beam")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("training_curve.png", dpi=150); plt.close()

# ---- figure 2: beam pattern for one example user (polar) ----
user = 100
phi_user = np.arctan2(pos[user, 0], pos[user, 1])     # true direction of the user
theta_ai = model(pos_t[user:user + 1]).detach().cpu().numpy()[0]
f_ai = np.exp(1j * theta_ai) / np.sqrt(M)
f_opt = np.exp(1j * np.angle(h_n[user])) / np.sqrt(M)  # ideal phase-conjugate beam

phis = np.linspace(-np.pi / 2, np.pi / 2, 721)
S = np.exp(1j * np.pi * np.outer(np.sin(phis), np.arange(M)))   # steering vectors
patt_ai = np.abs(S.conj() @ f_ai) ** 2
patt_opt = np.abs(S.conj() @ f_opt) ** 2
norm = patt_opt.max()

ax = plt.subplot(111, projection="polar")
ax.plot(phis, patt_ai / norm, label="AI beam")
ax.plot(phis, patt_opt / norm, ls="--", label="ideal beam")
ax.axvline(phi_user, color="k", ls=":", label="user direction")
ax.set_thetamin(-90); ax.set_thetamax(90); ax.set_theta_zero_location("N")
ax.set_title("Beam shape: AI vs ideal (one user)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))
plt.tight_layout(); plt.savefig("beam_pattern.png", dpi=150); plt.close()

# ---- figure 3: bar chart AI vs baselines ----
mrt = capacity((np.abs(h_va) ** 2).sum(1), SNR_DB).mean()
plt.figure(figsize=(6, 4))
names = ["DFT codebook\n(old method)", "AI model\n(ours)", "best possible\n(ceiling)"]
vals = [floor, rates[-1], ceiling]
colors = ["#d9534f", "#0275d8", "#5cb85c"]
plt.bar(names, vals, color=colors)
for i, v in enumerate(vals):
    plt.text(i, v + 0.05, f"{v:.2f}", ha="center")
plt.ylabel("spectral efficiency (bits/s/Hz)")
plt.title("AI vs baselines")
plt.tight_layout(); plt.savefig("comparison_bar.png", dpi=150); plt.close()

print("saved: training_curve.png, beam_pattern.png, comparison_bar.png")
