# Friday Team Sessions — the 2-hour weekly playbook

This is what the three of you do **together every Friday** (2 hours). The goal of
Friday is always the same: **merge everyone's week into one working thing and
produce the single Sunday demo.** You never debug live in front of the mentor.

You are Member 2 (DL engineer). Your job on Fridays: you own the code that ties
everyone's pieces together (`train.py`), so you usually drive the laptop.

---

## The Friday Ritual (same structure every week)

Do this in three blocks. Set a timer.

### Block 1 — MERGE (first 30 min)
- Everyone pushes their week's code to the **same GitHub repo**, one shared branch.
- On the shared Colab notebook: `!git clone ...` (or `git pull`) to get everyone's latest.
- Fix any conflicts together. Get it to the point where the notebook runs top-to-bottom with no red errors.
- **Rule:** if someone's code isn't ready, use last week's version or a dummy stand-in. Never let one person's delay block the merge.

### Block 2 — INTEGRATE & RUN (middle 60 min)
- Run the full pipeline end-to-end on Colab's GPU: data -> model -> loss -> (from Week 4) backward.
- Produce **this week's one deliverable** (the "Sunday demo" — see the per-week list below).
- Member 3 takes Member 1's data + your code to generate the week's plot/graph. You make sure it runs.

### Block 3 — REHEARSE & ASSIGN (last 30 min)
- Do a dry run of the Sunday demo. Whoever presents each part practices their 2-3 lines.
- Write down the **2-3 roadblocks** to ask the mentor (be specific, e.g. "we saw vanishing gradients in the loss").
- Assign next week's solo tasks to each person. Everyone leaves knowing exactly what to do Mon-Thu.

**End-of-Friday checklist:**
- [ ] Notebook runs clean on GitHub's latest.
- [ ] This week's graph/output is saved.
- [ ] Sunday demo rehearsed once.
- [ ] Next week's tasks assigned.

---

## Per-week Friday goals (what to actually produce)

> You've already done Weeks 1-3. Those Friday goals are listed for reference /
> in case you need to redo them; your live focus starts at **Week 4**.

### Week 1 Friday — Agree the ground rules *(done — retro-check)*
- **Together:** agree the notation everyone uses: `M` = antennas, `N` = users, `h` = channel, `f` = beamformer, `theta` = your model's phase output. Write it in the GitHub README ("shapes contract").
- **You bring:** the working GPU + dummy MLP.
- **Sunday demo:** run the sandbox script live, show the GPU.

### Week 2 Friday — Lock the data shapes *(done — retro-check)*
- **Together:** confirm the phase rule `f = exp(j*theta)/sqrt(M)` and unit-test that every `|f_i|` comes out equal (constant modulus). Member 3 leads the check, you code it.
- **You bring:** the DataLoader printing correct shapes.
- **From Member 1:** the real (or dummy) `.npy` files + the final value of M.
- **Sunday demo:** live DataLoader shape printout.

### Week 3 Friday — First integration *(done — retro-check)*
- **Together:** plug Member 3's spectral-efficiency function onto your model's output. Confirm it returns a number with **no autograd error**.
- **You bring:** `BeamNet` doing a clean forward pass.
- **Sunday demo:** forward pass output + GPU memory.

### Week 4 Friday — THE GRADIENT MILESTONE (your big one)
- **Together:** run `loss.backward()` on the fully integrated model and prove gradients flow from the physics loss back to the weights, and one step lowers the loss. **Rehearse this demo twice** — it's the most important checkpoint in the project.
- **You bring:** the training-loop skeleton (`train.py`: config, seed, optimizer, checkpoint saving) + the gradient checks.
- **From Member 3:** their real loss function (swap out the stand-in). If they have `gradcheck`, run it together.
- **Sunday demo:** `loss.backward()` running live, gradient norms finite and non-zero, loss dropping after one step.

### Week 5 Friday — Proof of learning (overfit)
- **Together:** merge Member 1's DFT codebook baseline number onto the same plot as your overfit curve, so the mentor sees the "floor you must beat."
- **You bring:** the model overfitting 10 samples (gap-to-optimum near zero).
- **Say clearly:** the *loss* stays negative; the *gap to the optimum* is what goes to ~0.
- **Sunday demo:** TensorBoard curve of the gap collapsing on 10 samples.

### Week 6 Friday — First real training + beam shapes
- **Together:** look at the polar plots (AI beam vs the optimal MRT beam) for a few users. Discuss whether they match.
- **You bring:** the first full 50-100 epoch training run with train/val curves; export predicted beams (`.npy`) for Member 3 to plot.
- **From Member 1:** the DFT baseline number to compare against.
- **Sunday demo:** train/val curves + polar beam plots.

### Week 7 Friday — Scaling + failure analysis
- **Together:** interpret *why* the worst 5% of users fail (extreme non-line-of-sight, grid edges, weak signal). Member 3 writes it up.
- **You bring:** three trained models (M = 16, 32, 64) + a table: parameters, memory, time, final rate; export the worst-5% users for Member 3.
- **From Member 1:** the "capacity scales with M" theory graph.
- **Sunday demo:** the three-size comparison table + failure report.

### Week 8 Friday — The headline graph
- **Together:** build the flagship figure — Capacity vs SNR (-10 to 20 dB) overlaying **AI model, MRT upper bound, DFT lower bound**. Compute "quantization loss recovered = (AI - DFT) / (MRT - DFT)".
- **You bring:** `evaluate.py` producing the AI curve across SNRs (+ the quantized 3-bit / 5-bit variants).
- **Sunday demo:** the single publication-quality Capacity-vs-SNR figure. This is your paper's money shot.

### Week 9 Friday — Speed + honesty
- **Together:** frame the latency comparison honestly — report both raw compute time AND pilot overhead (the AI needs no beam search / pilots; the codebook does). Agree the wording so you don't overclaim.
- **You bring:** the latency table (AI inference vs DFT exhaustive search, with warm-up + proper timing).
- **From Member 1/3:** the user-trajectory data + phase-continuity plot.
- **Sunday demo:** latency table + smooth beam-tracking plot.

### Week 10 Friday — The golden-model test
- **Together:** run the frozen "golden model" on the unseen scenario and decide how to phrase the claim (if zero-shot is weak, agree to show a few-shot fine-tune curve instead — still a valid result). Freeze the model *before* this; no tuning on the test scenario.
- **You bring:** the tagged golden checkpoint (weights + config + normalization) + inference results on the new scenario.
- **Sunday demo:** a table of AI vs baselines on the unseen street.

### Week 11 Friday — Package everything
- **Together:** compile all high-res figures + math derivations into one place; do a final clean-machine reproduce run from just the README.
- **You bring:** the CLI (`predict.py`) working, pinned `requirements.txt`, finished README, the demo rehearsed.
- **Sunday demo:** mentor types random X, Y, Z coordinates -> your CLI instantly prints the AI beamformer.

---

## Two rules to protect every Friday
1. **Merge to GitHub first, always.** No sharing code over WhatsApp/Drive zips. One repo, one truth.
2. **Big files (data, checkpoints) go to Google Drive, not GitHub.** Code on GitHub, data on Drive, run on Colab.

## Your standing Friday responsibilities (Member 2)
- Own `train.py` and the Colab notebook that runs the pipeline.
- Make sure the week's demo runs from a clean clone before everyone leaves.
- Bring the training curves / metrics; hand Member 3 whatever `.npy` outputs they need for plots.
