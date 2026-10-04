# Weeks 1-3: Theory + Step-by-Step Guide (Beginner Friendly)

This guide teaches you the ideas first, then walks you through the code. Read a
"Theory" box, then do the "Steps" under it. Do everything in **one Google Colab
notebook** with the GPU turned on.

---

## 0. The big picture - what are we even building?

Imagine a 5G/6G cell tower that has not one antenna but a **row of many small
antennas** (an "array"). By sending the same signal from each antenna but with a
tiny time/phase delay on each one, the tower can make the signals **add up in one
direction** - like a flashlight beam of radio waves aimed straight at your phone.
This aiming is called **beamforming**, and the delays are called **phase shifts**.

The question: *what phase shift should each antenna use to aim the beam perfectly
at a given user?*

- The **old way (codebook):** keep a fixed menu of, say, 64 pre-made beam
  directions and pick the best one. Problem: the menu is coarse, so the beam is
  never aimed *exactly* right. That wasted signal is "quantization loss."
- **Our way (this project):** train a neural network to output the **exact
  continuous phase values** - no menu, no rounding. In theory, a perfectly aimed
  beam every time.

Your model takes a user's **position** (x, y, z) and outputs **M phase numbers**
(one per antenna). A physics formula then scores how good that beam is, and we
train the network to make the score as high as possible.

### The symbols you'll see everywhere (keep this handy)
| Symbol | Means | Shape |
|---|---|---|
| `M` | number of antennas on the array | a single number (e.g. 64) |
| `N` | number of users in the dataset | e.g. 100,000 |
| `h` | the **channel** - how the signal travels from the array to one user | (M,) complex |
| `theta` (θ) | the **phase shifts** your network outputs | (M,) real numbers |
| `f` | the **beamformer** - the actual complex weights we apply | (M,) complex |
| `gain` | how strong the beam is at the user | one number |

### Complex numbers in 60 seconds (you need this)
A complex number is `a + b*i`. Think of it as an arrow on a 2D plane:
- its **length** (magnitude) = signal strength,
- its **angle** (phase) = timing of the wave.

Radio channels are complex because a wave has both strength *and* timing. Neural
networks only understand plain real numbers, so we always split a complex number
into its two real parts: **real part** `a` and **imaginary part** `b`. That's why
you'll see `h_re` and `h_im` instead of one complex `h`.

The one physics fact that drives the whole project:
> The beam strength a user gets is `gain = |h^H f|^2`
> ("how well the beamformer `f` lines up with the channel `h`").
> To make it biggest, each antenna's phase should **cancel out** the channel's
> phase. That's literally what your network is learning to do.

---

## Watch-then-do schedule (CampusX PyTorch playlist)

Match these titles to your own playlist - the numbers are the usual CampusX order
but may shift by one or two, so go by the **title**. For each row: watch the video
first (1.5x is fine), then immediately do the task under it in this guide. You do
**not** need the whole playlist for Weeks 1-3 - only these.

| Order | Watch (CampusX video) | Then do (this guide) |
|---|---|---|
| 1 | "Introduction to PyTorch" (what & why) | nothing - just context |
| 2 | "Tensors in PyTorch" | Week 1, steps 1.2 + 1.3 |
| 3 | "PyTorch nn Module" (building models) | Week 1, step 1.4 (the dummy net) |
| 4 | "Building a Neural Network / ANN in PyTorch" | finish Week 1 |
| 5 | "Dataset and DataLoader in PyTorch" | all of Week 2 |
| 6 | "PyTorch Training Pipeline" (watch, don't code yet) | helps you read Week 3 + prepares Week 4 |
| 7 | re-use videos 3 + 4 above (nn Module / ANN) | Week 3 (write BeamNet, forward pass) |

Videos you can **skip for now**: autograd (save it for Week 4), CNN, transfer
learning, RNN/LSTM - this project doesn't use them yet.

Fast path if you're short on time today: watch **Tensors**, **nn Module**,
**Dataset & DataLoader** (the three core ones) and do the tasks as you go. You can
watch the rest tonight.

---

## WEEK 1 - Set up the tools and train a throwaway network

### Theory first

**What is PyTorch?** A Python library for building and training neural networks.
Its basic data object is a **tensor** - basically a NumPy array that (a) can live
on a GPU for speed and (b) can automatically compute gradients for learning.

**Why a GPU?** Training does millions of multiply-add operations. A GPU does
thousands of them at once, so it's 10-100x faster than a CPU. Colab gives you one
free.

**What is a neural network, concretely?** A stack of `Linear` layers (each does
`output = input * weights + bias`) with a non-linear function (like ReLU or SiLU)
between them. The weights start random and are nudged, step by step, to make the
output better.

**The training loop** - memorize this rhythm, every project uses it:
1. **Forward:** feed input through the network to get a prediction.
2. **Loss:** measure how wrong the prediction is (one number).
3. **Backward:** `loss.backward()` - PyTorch computes how to change each weight.
4. **Step:** `optimizer.step()` - actually nudge the weights.
5. **Zero:** `optimizer.zero_grad()` - clear gradients before the next round.

Watch CampusX videos **2 (Tensors), 5 (nn.Module), 7 (ANN)** - they explain
exactly these.

### Steps

**1.1 - Open Colab with a GPU.**
- Go to https://colab.research.google.com -> New notebook.
- Runtime -> Change runtime type -> Hardware accelerator = **T4 GPU** -> Save.

**1.2 - Check PyTorch sees the GPU.** Paste in a cell and run:
```python
import torch
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())   # must say True
print("GPU name:", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"
```
If it says `False`: Runtime wasn't set to GPU. Redo step 1.1.

**1.3 - Play with tensors** (so they're not scary):
```python
x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape, x.dtype)      # shape = (3,), dtype = float32
y = x.to(device)             # move it onto the GPU
print(y * 2)                 # math works just like NumPy
```

**1.4 - Build and train a dummy network** on random data. This proves your whole
toolchain works. Don't worry that the data is meaningless - we only care that the
loss number goes **down**:
```python
import torch.nn as nn

M = 64
net = nn.Sequential(
    nn.Linear(3, 256), nn.SiLU(),   # 3 inputs (x,y,z) -> 256 hidden
    nn.Linear(256, M)               # 256 -> 64 outputs
).to(device)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)
x = torch.randn(4096, 3, device=device)   # 4096 fake users, 3 numbers each
y = torch.randn(4096, M, device=device)   # fake targets

for step in range(300):
    opt.zero_grad()                 # 5. clear
    pred = net(x)                   # 1. forward
    loss = ((pred - y) ** 2).mean() # 2. loss (mean squared error)
    loss.backward()                 # 3. backward
    opt.step()                      # 4. step
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")
```

**Week 1 is done when:** it prints `True` for the GPU, and the loss clearly drops
over the 300 steps. Screenshot that for your mentor.

---

## WEEK 2 - Get the real data flowing in batches

### Theory first

**What is the data?** For each user we have:
- a **position** `(x, y, z)` - 3 numbers,
- a **channel** `h` - `M` complex numbers describing how the signal from each of
  the M antennas reaches that user.

**Why split complex into real + imaginary?** Neural nets can't eat complex
numbers. So one channel of `M` complex values becomes **two** arrays of `M` real
values: `h_re` and `h_im`. No information is lost.

**Why normalize?** Two problems if we don't:
- Real channel values are *tiny* (like 0.000001) because radio signals weaken a
  lot over distance. Tiny numbers make the gradients tiny, and the network barely
  learns. Fix: divide all channels by one constant so their average power is ~1.
- Positions might be like x = 85.3 meters. Networks learn best when inputs are
  roughly in the range -1 to 1. Fix: rescale positions into [-1, 1].
- **Save these scaling constants.** You computed them from training data, and
  you'll need the exact same ones later at test time and in your final app.

**Dataset vs DataLoader** (CampusX video 6):
- A **Dataset** knows how to hand over **one** sample (one user's position +
  channel).
- A **DataLoader** bundles many samples into a **batch** (e.g. 512 at a time) and
  feeds them to the network. Batches are how we train efficiently.

**Why a "spatial" train/val/test split?** We split data into: **train** (learn
from), **validation** (check we're not cheating), **test** (final grade). If we
split randomly, two users standing right next to each other - who have almost
identical channels - could land one in train and one in test. The model would
look great but only because it basically saw the answer. So we split by **blocks
of location** instead, keeping nearby users together.

### Steps

**2.1 - Get the data from Member 1** (message them). You need two files:
`positions.npy` shape `(N, 3)` and `channels.npy` shape `(N, M)` complex, plus the
value of **M**. If it's not ready, don't wait - make a stand-in with the right
shapes:
```python
import numpy as np
N, M = 5000, 64
positions = np.random.uniform(-50, 50, (N, 3)).astype(np.float32)
channels  = (np.random.randn(N, M) + 1j*np.random.randn(N, M)).astype(np.complex64)
np.save("positions.npy", positions)
np.save("channels.npy", channels)
print("saved dummy data")
```

**2.2 - Load it and split complex into real/imag:**
```python
import numpy as np, torch
pos = np.load("positions.npy")
h   = np.load("channels.npy")
print("positions", pos.shape, "| channels", h.shape, h.dtype)

h_re = torch.tensor(h.real, dtype=torch.float32)   # real part
h_im = torch.tensor(h.imag, dtype=torch.float32)   # imaginary part
pos_t = torch.tensor(pos, dtype=torch.float32)
```

**2.3 - Normalize** (and keep the constants):
```python
# one scale for all channels so average power ~ 1
chan_scale = (h_re**2 + h_im**2).mean().sqrt()
h_re = h_re / chan_scale
h_im = h_im / chan_scale
print("channel scale (SAVE THIS):", chan_scale.item())

# positions into [-1, 1], using the min/max of the data
pos_min = pos_t.min(0).values
pos_max = pos_t.max(0).values
pos_t = 2 * (pos_t - pos_min) / (pos_max - pos_min) - 1
print("position range now:", pos_t.min().item(), "to", pos_t.max().item())
```

**2.4 - Make the split** (simple block split for now):
```python
n = len(pos_t)
n_train = int(0.7 * n); n_val = int(0.15 * n)
idx = torch.arange(n)
train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train + n_val]
test_idx  = idx[n_train + n_val:]
print("train/val/test sizes:", len(train_idx), len(val_idx), len(test_idx))
```

**2.5 - Build the DataLoader and print a batch:**
```python
from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(pos_t[train_idx], h_re[train_idx], h_im[train_idx])
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

pos_b, hre_b, him_b = next(iter(train_loader))   # grab one batch
print("batch position:", pos_b.shape)   # (512, 3)
print("batch h_re:", hre_b.shape)       # (512, 64)
print("mean power of batch (~1 good):", (hre_b**2 + him_b**2).mean().item())
```

**Week 2 is done when:** the printed shapes match `(512, 3)` and `(512, 64)`, and
the mean power is around 1. That printout is your mentor demo.

> Windows/Colab tip: keep `DataLoader(..., num_workers=0)` for now. Workers can
> cause confusing errors for beginners; you don't need them for this data size.

---

## WEEK 3 - Build the model and push a batch through it

### Theory first

**What should the model do?** Take a position `(3 numbers)` and output `M` phase
values `theta`. So it's a function `(3) -> (M)`.

**Why no activation on the last layer?** Hidden layers use activations (SiLU) to
bend and shape the function. But the final output is a **phase**, an angle, which
can be any real number (we'll wrap it with sine/cosine later). If we squashed it
with, say, a sigmoid (0 to 1), we'd forbid valid angles. So the last layer is a
plain `Linear` with nothing after it.

**From theta to the actual beamformer.** Your network outputs `theta`. The real
beamformer is:
```
f = exp(j * theta) / sqrt(M)
```
Two things this does:
- `exp(j*theta)` turns each phase into a complex number **on the unit circle** -
  i.e. every antenna sends at full power, only the *timing* differs. This is the
  "constant modulus" rule real hardware needs (a phase shifter can delay a signal
  but can't amplify it).
- dividing by `sqrt(M)` keeps the total transmit power fixed at 1, so comparisons
  are fair.

Because of this construction, the power constraint is satisfied **automatically** -
you never need to clip or add penalties. That's an elegant part of the design.

**The score (loss) - a first look.** How good is a beam? Its **array gain**
`gain = |h^H f|^2`, and the **spectral efficiency** (data rate) is
`rate = log2(1 + SNR * gain)`. Bigger rate = better. To *train*, we minimize the
**negative** rate:
```
loss = -mean( log2(1 + SNR * gain) )
```
Member 3 owns this formula; you just call it. One thing that confuses everyone:
**this loss is a negative number and never reaches 0.** Lower (more negative) is
better. Don't panic when you see `-4.2` - that's correct.

**What is a "forward pass"?** Just running data through the model once to get an
output. No learning yet - that's Week 4. This week we only prove the model runs
and the shapes are right.

### Steps

**3.1 - Write the model:**
```python
import torch.nn as nn

class BeamNet(nn.Module):
    def __init__(self, in_dim=3, M=64, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, M)        # final layer: NO activation
        )
    def forward(self, x):
        return self.net(x)              # returns theta, shape (batch, M)

model = BeamNet(in_dim=3, M=64).to(device)
print(model)
print("total parameters:", sum(p.numel() for p in model.parameters()))
```

**3.2 - Run one batch through and check the output shape:**
```python
pos_b = pos_b.to(device)
theta = model(pos_b)
print("theta shape:", theta.shape)     # expect (512, 64)
if device == "cuda":
    print("peak GPU memory (MB):", torch.cuda.max_memory_allocated() / 1e6)
```

**3.3 - Define the loss and run it on the output** (use this stand-in until
Member 3 gives you the real one):
```python
def neg_rate_loss(h_re, h_im, theta, snr_db=10.0):
    M = theta.shape[-1]
    c, s = torch.cos(theta), torch.sin(theta)        # exp(j*theta) = cos + j sin
    re = (h_re * c + h_im * s).sum(-1) / M**0.5       # real part of h^H f
    im = (h_re * s - h_im * c).sum(-1) / M**0.5       # imag part of h^H f
    gain = re**2 + im**2                              # |h^H f|^2
    snr = 10 ** (snr_db / 10)
    return -torch.log2(1 + snr * gain).mean()

loss = neg_rate_loss(hre_b.to(device), him_b.to(device), theta)
print("loss value:", loss.item())      # a NEGATIVE number is correct
```

**3.4 - Sneak peek at Week 4** - confirm it can go backward with no error:
```python
loss.backward()
first_layer = model.net[0]
print("gradient exists:", first_layer.weight.grad is not None)   # True = success
```

**Week 3 is done when:** `theta` is shape `(512, 64)`, the loss prints a
(negative) number, and `loss.backward()` runs with no error. Show the forward
output + memory + loss value to your mentor.

---

## Common beginner errors (and fixes)
| Error message / symptom | What it means | Fix |
|---|---|---|
| `CUDA not available` / trains on CPU | GPU runtime not on | Runtime -> change runtime type -> T4 GPU |
| `Expected all tensors on the same device` | some data on CPU, some on GPU | add `.to(device)` to the data too |
| `mat1 and mat2 shapes cannot be multiplied` | layer input size doesn't match | check `Linear(in, out)` numbers match your data |
| loss is `nan` | numbers blew up (often tiny/huge channels) | check your normalization (Week 2.3) |
| loss is a big negative number | that's normal here | not an error - lower is better |
| `RuntimeError: ... grad` twice | called `backward()` twice without re-running forward | run the forward pass again before each backward |

## What you can honestly tell the mentor after Week 3
"We have the full pipeline working end-to-end on dummy/real data: the DataLoader
feeds batches, BeamNet turns positions into continuous phases, and the
spectral-efficiency loss computes a value that we can already back-propagate
through. Next week we make it actually learn."

## Glossary (one line each)
- **Antenna array (M):** a row of antennas that together aim a beam.
- **Beamforming:** choosing phase shifts so the beam points at the user.
- **Channel (h):** complex numbers describing how the signal reaches the user.
- **Beamformer (f):** the complex weights we apply; `f = exp(j*theta)/sqrt(M)`.
- **Phase / theta:** the angle (timing) each antenna uses - your model's output.
- **Constant modulus:** every antenna at equal power; only phase changes.
- **Array gain:** how strong the beam ends up at the user, `|h^H f|^2`.
- **Spectral efficiency / rate:** data rate, `log2(1 + SNR*gain)`.
- **Codebook:** the old fixed menu of beams we're trying to beat.
- **Forward pass:** running data through the model to get an output.
- **Gradient / backward:** how the model figures out which way to adjust weights.
