# Member 2 - DL Engineer
# Week 3 task: the model itself. It takes a position (3 numbers) and outputs
# M phase values (theta) - one aim-setting per antenna.
#
# Important detail: the LAST layer has no activation. A phase can be any value
# (it gets wrapped by sin/cos later), so we must not squash it.

import torch
import torch.nn as nn


class BeamNet(nn.Module):
    def __init__(self, in_dim=3, M=64, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, M)        # output = theta, no activation here
        )

    def forward(self, x):
        return self.net(x)               # shape (batch, M)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BeamNet(in_dim=3, M=64).to(device)
    print(model)
    print("total parameters:", sum(p.numel() for p in model.parameters()))

    # push one fake batch through to check the output shape
    dummy = torch.randn(512, 3, device=device)
    theta = model(dummy)
    print("theta shape:", tuple(theta.shape))          # (512, 64)
    if device == "cuda":
        print("peak GPU memory (MB):", torch.cuda.max_memory_allocated() / 1e6)
