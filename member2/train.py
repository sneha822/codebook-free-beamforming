# Member 2 - DL Engineer
# Week 4 task: THE big milestone. Put everything together and prove the model
# can actually learn - that gradients flow from Member 3's physics loss all the
# way back to the network weights, and that one training step lowers the loss.
#
# This file is the "glue": it uses Member 1's data, Member 2's model, and
# Member 3's loss together.

import os, sys
import torch

# let this file import from the other members' folders
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "member3"))
from dataset import build_loaders          # our own (Member 2)
from model import BeamNet                  # our own (Member 2)
from loss import neg_rate_loss, array_gain, optimal_gain, rate_from_gain  # Member 3


def gradient_check(model, loss):
    # every weight must get a real (finite, non-zero) gradient, or learning
    # is silently broken somewhere.
    loss.backward()
    bad = []
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            bad.append(name)
        else:
            total += p.grad.norm().item()
    print("layers with missing / NaN gradient:", bad if bad else "none (good)")
    print("total gradient norm:", round(total, 4))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("device:", device)

    loaders, norm = build_loaders(batch_size=4096)
    model = BeamNet(in_dim=3, M=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ---- the Week 4 proof, on a single batch ----
    pos, h_re, h_im = next(iter(loaders["train"]))
    pos, h_re, h_im = pos.to(device), h_re.to(device), h_im.to(device)

    loss = neg_rate_loss(h_re, h_im, model(pos), snr_db=10)
    print("\nstarting loss:", round(loss.item(), 4), "(negative is normal)")
    gradient_check(model, loss)

    # clip gradients so a huge step cannot blow up training
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()

    with torch.no_grad():
        after = neg_rate_loss(h_re, h_im, model(pos), 10).item()
    print("loss before:", round(loss.item(), 4), "-> after one step:", round(after, 4),
          "(lower = learning works)")

    # ---- a short real training run so we see it improve over time ----
    print("\nshort training run:")
    for epoch in range(5):
        model.train()
        for pos, h_re, h_im in loaders["train"]:
            pos, h_re, h_im = pos.to(device), h_re.to(device), h_im.to(device)
            opt.zero_grad()
            loss = neg_rate_loss(h_re, h_im, model(pos), 10)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # check on validation data: compare our rate to the best possible rate
        model.eval()
        with torch.no_grad():
            pos, h_re, h_im = next(iter(loaders["val"]))
            pos, h_re, h_im = pos.to(device), h_re.to(device), h_im.to(device)
            theta = model(pos)
            rate = rate_from_gain(array_gain(h_re, h_im, theta), 10).mean().item()
            best = rate_from_gain(optimal_gain(h_re, h_im), 10).mean().item()
        print(f"  epoch {epoch}: our rate {rate:.3f} / best possible {best:.3f} "
              f"(gap {best-rate:.3f})")


if __name__ == "__main__":
    main()
