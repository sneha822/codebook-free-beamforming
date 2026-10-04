# My Learning + Task Tracker (Member 2 - DL Engineer)

How to use this file: tick `[ ]` -> `[x]` as you finish each item. Two parts:
- **Part A** = what I'm learning from the CampusX PyTorch playlist.
- **Part B** = the actual project tasks for Week 1, 2, 3 (all to be done TODAY).

---

# PART A - CampusX PyTorch Playlist Tracker

The CampusX PyTorch playlist teaches these topics in roughly this order. Match
each row to the video number in your own playlist. "Need for project" tells you
if this project actually uses it.

| # | Topic (CampusX) | Watched | Notes made | Need for project? |
|---|---|---|---|---|
| 1 | Introduction to PyTorch (what & why) | [ ] | [ ] | Yes - basics |
| 2 | Tensors in PyTorch (create, shape, dtype, `.to(device)`) | [ ] | [ ] | **Yes - core** |
| 3 | Autograd (automatic differentiation) | [ ] | [ ] | **Yes - the loss trains through this** |
| 4 | Training Pipeline in PyTorch | [ ] | [ ] | **Yes - your training loop** |
| 5 | `nn.Module` (building models) | [ ] | [ ] | **Yes - your model** |
| 6 | Dataset & DataLoader | [ ] | [ ] | **Yes - your data pipeline** |
| 7 | Building an ANN (first neural net) | [ ] | [ ] | **Yes - same idea as your model** |
| 8 | Training a NN on GPU | [ ] | [ ] | Yes - you train on Colab GPU |
| 9 | Improving / tuning the NN (optimizers, LR) | [ ] | [ ] | Yes - Weeks 5-6 |
| 10 | CNN | [ ] | [ ] | No - skip for now |
| 11 | Transfer Learning | [ ] | [ ] | No - skip for now |
| 12 | RNN / LSTM | [ ] | [ ] | No - skip for now |

**For TODAY (Weeks 1-3) the must-watch videos are: 2, 5, 6, 7** (and skim 8 for
GPU). Watch at 1.5x, don't take perfect notes - just enough to follow along.
Autograd (3) and Training Pipeline (4) you'll want before Week 4.

**One-line summary of each concept (so it sticks):**
- Tensor = a numpy array that can live on the GPU and remember its gradients.
- `nn.Module` = the box that holds your layers; you define `__init__` (layers) and `forward` (how data flows).
- Dataset = knows how to fetch ONE sample. DataLoader = bundles samples into batches for you.
- Autograd = PyTorch auto-computes gradients when you call `loss.backward()`.
- Training loop = repeat: forward -> loss -> `backward()` -> `optimizer.step()` -> `zero_grad()`.

---

# PART B - TODAY'S PLAN (Weeks 1, 2, 3)

You are doing 3 weeks in one day. That's fine - Weeks 1-3 are the light weeks.
Total time: about 3-5 hours. Do it all in **one Google Colab notebook** (free GPU).

**Before you start - open Colab and turn on the GPU:**
1. Go to https://colab.research.google.com -> New notebook.
2. Menu: Runtime -> Change runtime type -> Hardware accelerator = **T4 GPU** -> Save.

---

## WEEK 1 - "The Sandbox" (get PyTorch + GPU working)

**Goal in plain words:** prove PyTorch runs and sees the GPU, and train a tiny
throwaway neural net so you know the tools work.

- [ ] **Step 1.1** - In a Colab cell, run this and confirm it prints `True`:
  ```python
  import torch
  print(torch.__version__, torch.cuda.is_available())
  print(torch.cuda.get_device_name(0))   # should print "Tesla T4" or similar
  ```
- [ ] **Step 1.2** - Watch CampusX video 2 (Tensors). Then play: make a tensor, check `.shape`, move it to GPU with `.to("cuda")`.
- [ ] **Step 1.3** - Watch CampusX videos 5 + 7 (nn.Module, ANN). Then train a dummy net on random data:
  ```python
  import torch, torch.nn as nn
  device = "cuda"
  net = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, 64)).to(device)
  opt = torch.optim.Adam(net.parameters(), 1e-3)
  x = torch.randn(4096, 3, device=device)
  y = torch.randn(4096, 64, device=device)
  for step in range(200):
      opt.zero_grad()
      loss = ((net(x) - y)**2).mean()
      loss.backward()
      opt.step()
      if step % 50 == 0: print(step, loss.item())
  ```
- [ ] **Step 1.4** - Take a screenshot of the GPU name + the loss going down. That's your Week 1 proof for the mentor.

**What you need from your group for Week 1:** *(nothing blocking)*
- Just agree together on notation so all 3 of you use the same words:
  `M` = number of antennas, `h` = channel, `f` = beamformer, `theta` = the phases your net outputs. Write it in your shared GitHub README.

---

## WEEK 2 - "The Data Pipeline" (get data into a DataLoader)

**Goal in plain words:** load the antenna-channel data and feed it to your model
in batches, printing the shapes to prove it works.

**IMPORTANT - what you need from Member 1 (Wireless):** the data. Send this exact message now:

> "Hey, for the data pipeline I need two files:
> - `positions.npy` -> shape **(N, 3)**, each row is a user's x, y, z.
> - `channels.npy` -> shape **(N, M)**, complex numbers, the channel from the antenna array to each user.
> If the real DeepMIMO data isn't ready yet, please send me a **tiny dummy file with the correct shapes** so I'm not blocked. Also tell me the value of **M** (16, 32, or 64?)."

- [ ] **Step 2.1** - Watch CampusX video 6 (Dataset & DataLoader).
- [ ] **Step 2.2 (if Member 1 has NOT sent data yet - don't wait, use this):** make a fake array with the right shapes so you can keep going today:
  ```python
  import numpy as np
  N, M = 5000, 64
  positions = np.random.uniform(-50, 50, (N, 3)).astype(np.float32)
  channels  = (np.random.randn(N, M) + 1j*np.random.randn(N, M)).astype(np.complex64)
  np.save("positions.npy", positions); np.save("channels.npy", channels)
  ```
- [ ] **Step 2.3** - Load the files and split complex channels into real + imaginary (neural nets can't take complex numbers directly):
  ```python
  import numpy as np, torch
  from torch.utils.data import TensorDataset, DataLoader
  pos = np.load("positions.npy"); h = np.load("channels.npy")
  pos_t = torch.tensor(pos)
  h_re  = torch.tensor(h.real.astype(np.float32))
  h_im  = torch.tensor(h.imag.astype(np.float32))
  ds = TensorDataset(pos_t, h_re, h_im)
  loader = DataLoader(ds, batch_size=512, shuffle=True)
  ```
- [ ] **Step 2.4** - Print the shapes of one batch (this is your Week 2 proof):
  ```python
  p, hr, hi = next(iter(loader))
  print("pos", p.shape, "| h_re", hr.shape, "| h_im", hi.shape)
  ```
- [ ] **Step 2.5 (normalization - important, ask Member 3 to sanity-check):** real channel numbers are tiny, which stops the net from learning. Scale them:
  ```python
  scale = float((h_re**2 + h_im**2).mean().sqrt())
  h_re, h_im = h_re/scale, h_im/scale   # now mean power ~1
  print("scale constant (save this!):", scale)
  ```

**What you need from your group for Week 2:**
- **From Member 1:** the two `.npy` files + the value of M (see message above).
- **From Member 3:** confirm the rule "network outputs real numbers `theta`, and the beamformer is `f = exp(j*theta)/sqrt(M)`" - this is how your real outputs become valid complex phases. You'll use it in Week 3.

---

## WEEK 3 - "The Forward Pass" (build the real model, push a batch through)

**Goal in plain words:** write your actual model (`BeamNet`) that takes a
position and outputs `M` phase values, and pass one batch through it without crashing.

- [ ] **Step 3.1** - Write the model. Notice the last layer has **no activation** (phases can be any value):
  ```python
  import torch, torch.nn as nn
  class BeamNet(nn.Module):
      def __init__(self, in_dim=3, M=64, hidden=512):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(in_dim, hidden), nn.SiLU(),
              nn.Linear(hidden, hidden), nn.SiLU(),
              nn.Linear(hidden, M))          # outputs theta, no activation
      def forward(self, x):
          return self.net(x)                 # shape (batch, M)
  ```
- [ ] **Step 3.2** - Run one batch through and check the output shape is `(batch, M)`:
  ```python
  model = BeamNet(in_dim=3, M=64).to("cuda")
  theta = model(p.to("cuda"))     # p from Week 2
  print("theta shape:", theta.shape)
  print("peak GPU memory MB:", torch.cuda.max_memory_allocated()/1e6)
  ```
- [ ] **Step 3.3** - Attach Member 3's spectral-efficiency function to the output and confirm it returns a number with no error. Ask Member 3 for it; if they don't have it yet, use this standard version so you're not blocked:
  ```python
  def neg_rate_loss(h_re, h_im, theta, snr_db=10.0):
      M = theta.shape[-1]
      c, s = torch.cos(theta), torch.sin(theta)
      re = (h_re*c + h_im*s).sum(-1) / M**0.5
      im = (h_re*s - h_im*c).sum(-1) / M**0.5
      gain = re**2 + im**2
      snr = 10**(snr_db/10)
      return -torch.log2(1 + snr*gain).mean()

  loss = neg_rate_loss(h_re.to("cuda"), h_im.to("cuda"), theta)
  print("loss value:", loss.item())   # a negative number is CORRECT
  ```
- [ ] **Step 3.4** - Confirm it can go backward (a sneak peek at Week 4):
  ```python
  loss.backward()
  print("gradient exists:", model.net[0].weight.grad is not None)  # True = success
  ```

**What you need from your group for Week 3:**
- **From Member 3:** the spectral-efficiency (SE) loss function. If it's not ready, use the version above and tell Member 3 you did, so they can verify the math later.
- **From Member 1:** confirm the final value of **M** so your model's output size is correct.

---

## END OF TODAY - checklist before you close the laptop

- [ ] Colab notebook runs top-to-bottom with no red errors.
- [ ] You can print: GPU name, one batch's shapes, model output shape, and a loss value.
- [ ] Save the notebook to your shared GitHub / Google Drive.
- [ ] Message your group: "Weeks 1-3 done on my side (using dummy data where needed). Member 1, I still need the real .npy files + final M. Member 3, please double-check the loss function I used."

## Honest thing to tell your mentor
You merged Weeks 1-3 into one day and used stand-in (dummy) data where the real
DeepMIMO data wasn't ready. That's normal engineering. The pipeline is proven;
swapping in real data later changes nothing else.

---

*Tip: keep a tiny `LOG.md` too - one line per day: what I did, what broke, what I
learned. It becomes your thesis "methodology" section for free.*
