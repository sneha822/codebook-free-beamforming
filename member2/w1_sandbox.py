# Member 2 - DL Engineer
# Week 1 task: prove PyTorch works and the GPU is being used, by training a
# throwaway network on random data. The data is meaningless - we only care that
# the loss goes DOWN, which proves the whole training machinery works.

import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# a tiny network: 3 inputs (x, y, z) -> 256 hidden -> 64 outputs
net = nn.Sequential(
    nn.Linear(3, 256), nn.SiLU(),
    nn.Linear(256, 64)
).to(device)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# fake data
x = torch.randn(4096, 3, device=device)
y = torch.randn(4096, 64, device=device)

for step in range(300):
    opt.zero_grad()
    pred = net(x)
    loss = ((pred - y) ** 2).mean()   # mean squared error
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

print("done - if the loss dropped, the setup works")
