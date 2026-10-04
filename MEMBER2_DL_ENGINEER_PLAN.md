# Member 2 (DL Engineer) - 11-Week Task Plan

Project: Codebook-free continuous phase-shift predictor, trained with a direct spectral-efficiency loss on DeepMIMO (O1).
Your job: data loading into PyTorch, model architecture, training loop, experiments, latency, CLI. Member 1 supplies the data, Member 3 supplies the math (loss, baselines, analysis).

---

## 0. How your week works

| Day | Time | What |
|---|---|---|
| Mon, Tue, Wed, Thu | 1 h each | Solo: 15 min study + 45 min build |
| Fri | 2 h | Joint team session: merge into ONE branch, run everything end to end, no live debugging on Sunday |
| Sat | 1 h | Solo: polish the demo script, prepare your 3-4 minute update |
| Sun | 30-45 min | Mentor meeting |

Rules for yourself:
- Commit every day, even if small. Keep a `LOG.md` (date, what I did, what broke, what I learned). This becomes your thesis methodology section later.
- Every deliverable is a **script that runs from a clean terminal** (`python scripts/xxx.py`), not a notebook cell.
- The mentor wants a live run in the first 15 minutes. Each week's "Sunday demo" below is one command.

## 1. This week (Sep 29 - Oct 4) - you are already in Week 3 by the calendar

Week 1 showcase was Sep 20, Week 2 was Sep 27, Week 3 is Oct 4. If you and the team have not done Weeks 1-2 yet, do this compressed version. It is about 6 hours total, so it takes every solo day plus the Friday session.

| Day | Task |
|---|---|
| Tue (today) | Do Week 1 Mon+Tue: environment, GPU check, dummy MLP on GPU (see the Appendix A snippet) |
| Wed | Do Week 1 Wed+Thu: training loop on random data, TensorBoard, repo skeleton pushed to a shared GitHub repo |
| Thu | Do Week 2: get a real (or fake, same-shape) `(N, M)` complex array from Member 1 and build `dataset.py` with the DataLoader shape printout |
| Fri (joint) | Write `model.py` (Week 3) and pass one batch through it. Agree on the shapes contract (section 2) with the team |
| Sat | Polish demo: one script that prints GPU, tensor shapes, and the forward-pass output shape |
| Sun Oct 4 | Show all three deliverables at once and be honest that Weeks 1-2 were merged into this one |

Tell your mentor the true status. It is far better than pretending.

## 2. Decisions to make in the first team session (important)

**Decision A - What is the model's input? (This affects Week 10 and Week 11.)**
- Your Week 11 CLI takes X, Y, Z coordinates, so the primary model is position -> phases.
- But a position-input MLP **cannot zero-shot generalize to a different street (O2)**, because the map from coordinates to channel is specific to the O1 geometry. It memorizes O1. That contradicts the Week 10 goal.
- Recommended: build **two input modes** behind one config flag:
  1. `input=position` (x, y, z, optionally BS-relative) -> used for the CLI and the O1 results.
  2. `input=channel` (real and imaginary parts of a noisy or partial channel estimate, size 2M) -> used for the O2 generalization test.
- Take this to the mentor in the Week 2 or 3 meeting as a roadblock question. It is exactly the kind of specific question they want.

**Decision B - Verify O2 exists.** I am not certain DeepMIMO ships a scenario called "O2" (it has O1 at 28/60/140 GHz, I1, I3, Boston5G, ASU campus, and others). Member 1 should check in Week 1. If O2 does not exist, the fallback is another scenario, or a held-out spatial region and a different BS in O1. Decide this early because the array geometry (M) has to match.

**Decision C - Shapes contract** (write it in `README.md`, all three of you sign off):
- `h`: `(N, M)` complex64, single-antenna user, single subcarrier, channel from BS to user.
- Network output: `theta` shape `(B, M)`, real float32, raw and unconstrained.
- Beamformer: `f = exp(j*theta) / sqrt(M)`, so `|f_i| = 1/sqrt(M)` always and total power = 1. No clipping and no penalty terms are needed because the constraint is satisfied by construction.
- Loss convention: use `h^H f` consistently (the pasted formula says `h^T f`; the conjugate convention only changes which side is conjugated. Agree once, write it down).

**Decision D - Split the data spatially, not randomly.** Neighbouring grid points have nearly identical channels, so a random split leaks validation into train and your numbers look better than they are. Split by contiguous blocks (for example DeepMIMO user rows) and Member 1 should record which rows go where.

**Practical warning:** your project folder is inside OneDrive. Keep the DeepMIMO data, `.venv`, checkpoints and `runs/` outside OneDrive (for example `C:\dev\beam\`) or OneDrive will try to sync gigabytes and can corrupt files. Keep only code in the OneDrive folder and in git.

## 3. Repo layout (create this in Week 1)

```
beam/
  README.md
  LOG.md
  requirements.txt
  configs/         base.yaml, M16.yaml, M32.yaml, M64.yaml
  src/
    data.py        Dataset, DataLoader, normalization
    model.py       BeamNet
    loss.py        (Member 3 writes, you import) spectral-efficiency loss
    baselines.py   (Members 1+3) MRT, DFT codebook
    train.py
    evaluate.py
    predict.py     CLI (Week 11)
  scripts/         one runnable demo per week
  tests/           shape tests, constant-modulus test, gradient test
```

## 4. Study resources (all free)

- PyTorch official "Learn the Basics" tutorial series (tensors, Dataset/DataLoader, autograd, optimization). Weeks 1-4.
- Andrej Karpathy, "A Recipe for Training Neural Networks" (blog) - the single best guide to debugging training. Week 5.
- Andrej Karpathy, "Neural Networks: Zero to Hero" (YouTube) - watch the first two videos (backprop, makemore MLP). Weeks 3-5.
- DeepMIMO docs at deepmimo.net and the DeepMIMO paper (Alkhateeb, 2019). Weeks 1-2, shared with Member 1.
- Alkhateeb et al., "Deep Learning Coordinated Beamforming for Highly-Mobile Millimeter Wave Systems" (IEEE Access, 2018). Background for the problem and the codebook baseline.
- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (NeurIPS 2020). Read in Week 6. Coordinates-to-beam is a high-frequency function, so a plain MLP will underfit without something like this.
- Search terms for related work (for the paper later): "unsupervised learning beamforming sum rate loss", "deep learning direct hybrid precoding mmWave".

---

## 5. Week-by-week tasks

### Week 1 (showcase Sep 20) - The Engineering Sandbox
Goal: a working GPU environment and a dummy MLP.
- **Mon:** Install Python 3.10 or 3.11, create a venv (outside OneDrive), install the PyTorch build that matches your CUDA from pytorch.org. Run Appendix A. If `cuda.is_available()` is False (no NVIDIA GPU), set up Google Colab or Kaggle notebooks as the shared GPU and tell the team now. Study: tensors, dtypes, `.to(device)`.
- **Tue:** Study autograd and `nn.Module`. Write a dummy MLP (`in=3 -> 256 -> 256 -> M`).
- **Wed:** Write a training loop on random data with a made-up loss. Time it on CPU vs GPU.
- **Thu:** Log the loss with `torch.utils.tensorboard.SummaryWriter`. Create the git repo and the layout in section 3.
- **Sat:** Write `scripts/w1_sandbox.py`: prints GPU name and memory, trains the dummy MLP for 200 steps, writes a TensorBoard log.
- **Fri joint:** Agree notation (M, N, h, f, theta) and the shapes contract with Member 3 and Member 1.
- **Sunday demo:** `python scripts/w1_sandbox.py` live in the terminal.
- **Done when:** the script runs from a clean shell and shows GPU utilisation.

### Week 2 (showcase Sep 27) - The Data Pipeline
Goal: real DeepMIMO data flowing through a DataLoader.
- **Mon:** Study `Dataset`, `DataLoader`, `batch_size`, `shuffle`, `pin_memory`. Read how Member 1 extracts DeepMIMO arrays.
- **Tue:** Write `data.py`: load the `.npy` files (channels `(N, M)` complex64, positions `(N, 3)`), return real/imag stacked tensors.
- **Wed:** Normalization. Raw mmWave channel magnitudes are tiny (path loss can be around 1e-5 to 1e-9), which will make gradients vanish. Scale the channels by one global constant so the mean `|h|^2` is about 1 (or work in a scaled domain). Save that constant with the checkpoint; you will need it at inference and in the CLI. Normalize positions to [-1, 1] using train-set statistics only.
- **Thu:** Spatial train/val/test split (Decision D). Windows note: set `num_workers=0` first, and put DataLoader code under `if __name__ == "__main__":` before increasing it. The full dataset (100k x 64 complex) is only tens of MB, so just put it all on the GPU.
- **Sat:** `scripts/w2_dataloader.py` prints batch shapes, dtypes, min/max/mean of the normalized values. Add `tests/test_shapes.py`.
- **Fri joint:** With Member 3, check the phase mapping `theta -> f` gives `|f_i|` constant (a unit test).
- **Sunday demo:** live DataLoader printout.
- **Done when:** the printed shapes match the shapes contract exactly.

### Week 3 (showcase Oct 4) - The Forward Pass
Goal: `model.py` runs a batch through without crashing.
- **Mon:** Study MLP design choices: ReLU vs GELU/SiLU, LayerNorm vs BatchNorm, residual blocks, weight init.
- **Tue:** Write `BeamNet` (Appendix B): input encoder, N hidden blocks, linear head with **no activation** producing `theta`.
- **Wed:** Pass a large batch through (for example B=4096, M=64). Report `torch.cuda.max_memory_allocated()`. Try B=100k to know your limits.
- **Thu:** Make M, width, and depth config-driven (`configs/*.yaml`) so Week 7 is trivial.
- **Sat:** `scripts/w3_forward.py`: load a batch, forward pass, print `theta` shape, `|f|` constant-modulus check, and peak memory.
- **Fri joint:** Plug Member 3's dummy differentiable SE function onto the output and confirm no autograd error.
- **Sunday demo:** forward pass output plus memory report.
- **Done when:** end-to-end forward pass with no memory errors.

### Week 4 (showcase Oct 11) - Gradient Flow
Goal: `loss.backward()` works through the whole chain.
- **Mon:** Study how autograd handles complex numbers. Simplest and safest: avoid complex tensors in the graph, use the real/imag split in Appendix A of the loss. Then no complex-autograd surprises.
- **Tue:** Integrate the real loss from Member 3 with `BeamNet`. Run one forward/backward step.
- **Wed:** Gradient checks: every parameter has a non-None, finite, non-zero `.grad`. Also `torch.autograd.gradcheck` on the loss function alone in float64. Add `clip_grad_norm_`.
- **Thu:** Write the `train.py` skeleton: config loading, seed, optimizer (AdamW), checkpoint saving.
- **Sat:** `scripts/w4_backward.py`: prints loss, gradient norm per layer, and a check that a single SGD step lowers the loss on the same batch.
- **Fri joint:** Live-demo rehearsal. This is a critical milestone, so rehearse twice.
- **Sunday demo:** `python scripts/w4_backward.py`.
- **Done when:** gradient norms are finite and non-zero in every layer.

### Week 5 (showcase Oct 18) - Proof of Learning (overfit 10 samples)
Goal: the model can memorize 10 samples.
- **Mon:** Read Karpathy's "Recipe" (overfit one batch first).
- **Tue:** Write `scripts/w5_overfit.py`: 10 fixed samples, no shuffle, no regularization, 2000 steps, lr 1e-3.
- **Wed:** Fix problems: try lr 1e-2 to 1e-4; check normalization scale; check the loss is not saturated (if `SNR * gain` is huge, `log2(1+x)` gradients are tiny, so choose a sensible SNR scale, for example 10-20 dB on normalized channels).
- **Thu:** Log to TensorBoard: loss, and **gap to optimum**.
- **Sat:** Polish the curves.
- **Fri joint:** Merge Member 1's DFT baseline numbers into the same plot.
- **IMPORTANT CORRECTION to the roadmap:** the loss `-log2(1+SNR*gain)` does **not** go to zero. It converges to a negative number `-log2(1+SNR*G_opt)`. For a phase-only array with `h` known, the optimum is `theta_i = angle(h_i)` (phase-conjugate) with `G_opt = (sum_i |h_i|)^2 / M`. So plot the **gap = achievable_rate - model_rate**, which goes to ~0. Tell the mentor this up front so "near zero" is understood as "near the optimum".
- **Done when:** on the 10 samples the gap is below about 1% of the optimum rate.

### Week 6 (showcase Oct 25) - First Real Training
Goal: 50-100 epoch training run on 100k samples with train/val curves.
- **Mon:** Study LR schedules (cosine, ReduceLROnPlateau), AdamW weight decay, early stopping. Skim the Fourier Features paper.
- **Tue:** Complete `train.py`: epoch loop, validation each epoch, best-checkpoint saving, TensorBoard train/val loss, mean rate and mean gap to optimum.
- **Wed:** Run the full training. Save the config with every run (run folder name = date + config hash).
- **Thu:** If val is much worse than train, add regularization; if both are bad, the model underfits. Fix underfitting with Fourier features (`x -> [sin(2*pi*B*x), cos(2*pi*B*x)]`) in the input encoder, and a wider/deeper net.
- **Sat:** `scripts/dump_beams.py`: save predicted `f` for chosen users to `.npy` so Member 3 can make polar plots.
- **Fri joint:** Look at polar plots together, AI vs MRT.
- **Done when:** val gap curve is smooth and you can state the final mean rate vs the DFT baseline.

### Week 7 (showcase Nov 1) - Scaling M = 16, 32, 64
- **Mon:** Study how the parameter count scales with M (the head is `hidden x M`). Decide whether width should scale with M.
- **Tue-Wed:** Train three models with configs `M16/M32/M64` using the same seed and split. Record parameters, time per epoch, peak memory, final val rate.
- **Thu:** Comparison plot: curves for the three sizes, and a table (parameters, memory, time, rate, gap).
- **Sat:** `scripts/dump_worst.py`: write the worst 5% of test users (by gap) with position, `|h|` norm, LoS/NLoS tag if Member 1 has it, for Member 3's failure analysis.
- **Fri joint:** Interpret failures: for example extreme NLoS, edge of the grid, low `|h|`.
- **Done when:** three converged runs and a clean table.

### Week 8 (showcase Nov 8) - SNR Stress Test
- **Mon:** Study why the SNR in the loss matters little: for a single user, maximizing `log2(1+SNR*g)` is the same as maximizing `g` for any SNR, since `log` is monotonic. SNR only changes the weight each user gets in the average gradient (at high SNR, strong users saturate). Explain this to the mentor; it is a nice insight.
- **Tue:** `evaluate.py`: for SNR from -10 to 20 dB compute mean rate for AI, MRT (upper bound) and DFT (lower bound) on the test set.
- **Wed:** Optionally train three variants at SNR = 0, 10, 20 dB and compare to show the weighting effect; also try training with a random SNR each batch.
- **Thu:** Add the quantized phase-shifter variants (for example 3-bit and 5-bit rounding of the AI output) as extra curves. It shows what quantization costs and is a good paper figure.
- **Sat:** Publication-quality figure: vector PDF and 300 dpi PNG, labelled axes, readable at column width.
- **Fri joint:** Compute "quantization loss recovered" = (AI - DFT) / (MRT - DFT).
- **Done when:** one figure with AI, MRT and DFT overlays.

### Week 9 (showcase Nov 15) - Tracking and Latency
- **Mon:** Study how to benchmark correctly: warm-up runs, `torch.cuda.synchronize()` before and after timing, batch size 1 vs large batch, CPU vs GPU, median and p95 over many runs.
- **Tue:** `scripts/latency.py`: AI forward pass at batch=1 (CPU and GPU).
- **Wed:** Time the DFT exhaustive search on the same hardware: computing `|h^H f_k|^2` for all K codewords and taking the argmax.
- **Thu:** Trajectory inference: feed the ordered positions along Member 1's trajectory and record `theta` over time; give Member 3 the sequence to plot phase continuity.
- **Sat:** Latency table (mean, median, p95, hardware).
- **Fri joint:** Frame the comparison honestly. In software, the DFT search is also a fast matrix multiply, so a plain compute-time race may not favour the AI by much. The real advantage is in **over-the-air training overhead**: exhaustive beam search needs K pilot measurements, the position-based net needs none. Report both compute latency and pilot overhead.
- **Done when:** the table is reproducible with one command.

### Week 10 (showcase Nov 22) - Zero-Shot on Unseen Scenario
- **Mon:** Study domain shift and why position-input models do not transfer (Decision A).
- **Tue:** Freeze the golden model: checkpoint, config, normalization constants, git tag `golden-v1`. No more tuning after this. Tuning on the test scenario is not zero-shot.
- **Wed:** Run inference on the unseen scenario with the channel-input model, and also the position-input one, and record both. Make sure the array geometry (M) is identical.
- **Thu:** Compare to MRT and DFT there.
- **Sat:** Prepare a slide that reports the result honestly. If zero-shot is poor, add a few-shot fine-tune curve (fine-tune on 100 / 1000 samples) which is a legitimate research result too.
- **Fri joint:** Decide how to phrase the claim in the paper.
- **Done when:** a table with AI vs baselines on the unseen scenario.

### Week 11 (showcase Nov 29) - Production-Ready Package
- **Mon:** `src/predict.py` CLI with `argparse`: `python -m src.predict --x 120.5 --y 45.2 --z 1.5 --ckpt golden.pt` prints the phases (degrees) and the complex beamformer, and the predicted rate if a channel is available.
- **Tue:** Checkpoint contains model weights, config, and normalization constants; the CLI loads only that. Add input range checking and a clear error for out-of-coverage coordinates.
- **Wed:** `requirements.txt` with pinned versions, seeds set, `README.md` with quickstart, results table, and figure gallery.
- **Thu:** Reproduce the golden result on a clean machine or Colab from scratch using only the README.
- **Sat:** Rehearse the CLI live demo.
- **Fri joint:** Compile all figures and derivations.
- **Done when:** a stranger can clone, install, and get the same number.

---

## Appendix A - GPU check plus the loss (real/imag form, no complex autograd needed)

```python
import torch, time
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"

def neg_rate_loss(h_re, h_im, theta, snr_db):
    """h: (B, M) channel real/imag, theta: (B, M) phases from the network.
    f = exp(j*theta)/sqrt(M);  gain = |h^H f|^2."""
    M = theta.shape[-1]
    c, s = torch.cos(theta), torch.sin(theta)
    re = (h_re * c + h_im * s).sum(-1) / M ** 0.5
    im = (h_re * s - h_im * c).sum(-1) / M ** 0.5
    gain = re ** 2 + im ** 2
    snr = 10 ** (snr_db / 10)
    return -torch.log2(1 + snr * gain).mean()

def optimal_gain(h_re, h_im):
    """Best possible gain for a constant-modulus beamformer: (sum |h_i|)^2 / M."""
    mag = torch.sqrt(h_re ** 2 + h_im ** 2)
    return mag.sum(-1) ** 2 / h_re.shape[-1]
```

## Appendix B - BeamNet skeleton

```python
import torch, torch.nn as nn

class BeamNet(nn.Module):
    def __init__(self, in_dim, M, hidden=512, depth=4, fourier_feats=0, scale=10.0):
        super().__init__()
        self.M = M
        self.fourier_feats = fourier_feats
        if fourier_feats:
            self.register_buffer("B", torch.randn(in_dim, fourier_feats) * scale)
            in_dim = 2 * fourier_feats
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, M)      # raw phases, no activation

    def forward(self, x):
        if self.fourier_feats:
            p = 2 * torch.pi * x @ self.B
            x = torch.cat([torch.sin(p), torch.cos(p)], dim=-1)
        return self.head(self.body(x))         # theta, shape (B, M)
```
