# Project Brief — Leader's Reference

**Project:** Codebook-Free Continuous Phase-Shift Predictor using a Direct Array-Gain Loss
**Role of this doc:** everything the group leader needs to explain the project, the
division of work, and the key technical decisions to the mentor. Keep it open
during the meeting.

---

## 1. One-paragraph summary (the opener)

In mmWave/THz systems, a base station with a large antenna array must form a
narrow beam toward each user. Conventional 5G selects this beam from a discrete,
quantized **codebook**, which limits spatial resolution and leaves a permanent
"quantization loss." We build a **codebook-free** neural network that outputs
**continuous phase-shifter values** directly, trained end-to-end against a
**differentiable spectral-efficiency loss** grounded in the physical channel. No
codebook, no quantization — fine-grained beam alignment.

**The objective we optimize:**
```
L = - log2( 1 + SNR * | h^H f |^2 )
```
The network outputs phases theta; the beamformer is constructed as
`f = exp(j*theta) / sqrt(M)`, which enforces the **constant-modulus** constraint
by construction (unit-magnitude phase shifters, fixed total power).

**The one-sentence pitch:** "We've built a differentiable pipeline where a
physics-grounded spectral-efficiency loss trains a network to output continuous
beamformer phases directly — and we measure it against both the MRT upper bound
and the DFT codebook lower bound to quantify the quantization loss we recover."

---

## 2. End-to-end technical pipeline

1. **Channel data** — DeepMIMO ray-tracing (scenario O1): per user, position
   (x, y, z) and channel vector h in C^M.
2. **Preprocessing** — split complex h into real/imag; normalize channel power to
   ~1; normalize positions to [-1, 1]; spatial train/val/test split.
3. **Model (BeamNet)** — maps input -> theta in R^M (linear output head, no activation).
4. **Beamformer construction** — f = exp(j*theta)/sqrt(M).
5. **Loss** — negative spectral efficiency, differentiable, implemented in
   real/imag arithmetic to avoid complex-autograd issues.
6. **Training** — gradients flow from the physics loss back to real-valued
   weights; AdamW, gradient clipping.
7. **Evaluation** — compare vs MRT (upper bound) and DFT codebook (lower bound)
   across SNR; latency; generalization to an unseen scenario.

---

## 3. Division of work

### Member 1 — Wireless / Channel Lead
- **Dataset generation:** extract DeepMIMO O1 channels + positions; define array
  geometry (M = 16/32/64), carrier, subcarrier handling; deliver clean .npy
  arrays with a documented shapes contract.
- **Physical characterization:** channel-magnitude maps, LoS/NLoS labeling,
  dead-zone identification.
- **Baselines:** MRT (maximum-ratio transmission — capacity upper bound) and the
  DFT codebook (exhaustive-search lower bound). These define the performance envelope.
- **Scaling & mobility:** theoretical capacity-vs-M scaling; a user trajectory for
  the tracking test; sourcing the held-out scenario for the zero-shot test.

### Member 2 — Deep Learning Engineer (group leader)
- **Data pipeline:** Dataset/DataLoader, normalization (persisted constants),
  spatial splitting.
- **Model architecture:** BeamNet — MLP with a continuous phase head; configurable
  depth/width/M; Fourier-feature input encoding if the position->beam map underfits.
- **Training infrastructure:** full training loop, checkpointing, config
  management, TensorBoard logging, seeds, reproducibility.
- **Experiments:** overfit sanity check, full-scale training, M-sweep, SNR sweep,
  latency benchmarking, zero-shot/golden-model run.
- **Deliverable app:** CLI mapping input coordinates -> predicted complex beamformer.
- **Integration role:** train.py imports Member 1's data loader and Member 3's
  loss — owns the glue; runs the Friday merges.

### Member 3 — Mathematics / Signal Processing Lead
- **Loss design:** the differentiable spectral-efficiency objective; correct
  h^H f convention; numerically stable real/imag implementation; gradcheck validation.
- **Optimality analysis:** closed-form optimum theta_i = angle(h_i) with
  G_opt = (sum_i |h_i|)^2 / M — the reference the network is measured against.
- **Analysis & figures:** polar beam-pattern comparisons (AI vs MRT); failure
  analysis of worst-case users; the headline Capacity-vs-SNR figure; the
  quantization-loss-recovered metric (AI - DFT)/(MRT - DFT).

**Handoffs:** Member 1 -> (data, baselines) -> Member 2 -> (beam predictions) ->
Member 3 -> (loss, analysis, figures). The training loop cannot run without both
Member 1's data and Member 3's loss — hence the weekly Friday merge.

---

## 4. Key technical decisions to defend (mentor will probe these)

1. **"The loss doesn't go to zero."** Correct — it converges to a negative
   constant. The quantity that goes to ~0 is the **gap to the closed-form
   optimum**. We report the gap, not the raw loss.

2. **Input choice — position vs channel.** A position-input network memorizes the
   O1 geometry and cannot zero-shot transfer. So we support **two input modes**:
   position-input (CLI + O1 results) and channel-input (the generalization test).
   This resolves the tension between the CLI deliverable and the zero-shot claim.

3. **What's actually novel.** With a known channel, the phase-only optimum is
   closed-form (theta = angle(h)) — the network isn't discovering unknown physics.
   The real contribution is (a) predicting a near-optimal beam from **position
   alone, with no pilot/channel measurement** (which the pilot-hungry codebook
   search cannot do), and/or (b) **generalization** across geometry. We frame the
   claim around pilot/measurement overhead, not just "we removed the codebook."

4. **Latency framing.** In software the DFT search is also a fast matrix multiply,
   so a raw compute race is modest. The decisive advantage is **over-the-air
   overhead**: exhaustive search needs K pilot measurements; the position-based
   net needs none. We report both compute latency and pilot overhead.

5. **No data leakage.** Spatial (block) split, not random — adjacent users have
   near-identical channels, so random splitting inflates results.

---

## 5. Milestone arc (the narrative)

Environment -> data pipeline -> forward pass -> **gradient flow (pivotal: physics
loss back-propagating to real weights)** -> overfit proof -> full training ->
M-scaling -> SNR benchmark -> latency -> zero-shot generalization -> reproducible
package + CLI.

---

## 6. Quick-answer cheat sheet (if put on the spot)

- **"Why continuous phases?"** Removes codebook quantization; enables exact beam
  alignment instead of nearest-menu-entry.
- **"Why is the loss negative?"** It's negative spectral efficiency; we minimize
  it, so more negative = higher data rate.
- **"How do real outputs become valid complex phases?"** f = exp(j*theta)/sqrt(M);
  constant modulus and unit power hold by construction — no clipping or penalties.
- **"What's your baseline?"** MRT as the upper bound, DFT codebook as the lower
  bound; we report where the AI sits between them.
- **"How do you avoid complex-gradient problems?"** The loss is written entirely in
  real/imag arithmetic, so autograd only ever sees real numbers.
- **"What proves it generalized and didn't memorize?"** The zero-shot test on an
  unseen scenario with the frozen golden model.
