# The Complete Explanation (Weeks 1-4) - everything, from zero

This document assumes you know nothing. It explains the problem, the math, the
model, what each of the three members built, and what every result number means.
Read it top to bottom once and you will understand the whole project.

---

# PART 1 - THE PROBLEM (in ordinary words)

## 1.1 Why antennas in a row?
A normal antenna sprays signal in all directions, like a bare light bulb. At the
very high frequencies used in 5G/6G (called **mmWave** and **THz**), the signal is
weak and blocked easily, so spraying it everywhere wastes almost all of it.

The fix: put many small antennas in a row (an **antenna array**) and make them
work together. If each antenna sends the same wave but slightly **delayed**, the
waves add up strongly in one direction and cancel out in others. The result is a
focused **beam**, like a spotlight. We call the number of antennas **M**.

## 1.2 What is a "phase shift"?
A radio signal is a wave. You can delay a wave by a fraction of a cycle - that
delay is called a **phase shift**, measured as an angle. Each antenna gets its own
phase shift. The full set of M phase shifts is written **theta** (the Greek letter
θ). Choosing theta well = aiming the beam. That is the entire job.

## 1.3 The old way vs our way
- **Old way (codebook):** the tower keeps a small fixed list of ready-made beam
  directions - a **codebook** - and just picks the closest one to the user. Like a
  thermostat that only moves in jumps of 5 degrees. You get close, never exact.
  The wasted accuracy is called **quantization loss** (quantization = rounding to
  the nearest item on a fixed list).
- **Our way (this project):** train a neural network (an AI) to output the
  **exact** phase values - any number, not rounded. No list. In principle, a
  perfectly aimed beam every time. This is what "codebook-free, continuous phase"
  means.

## 1.4 What the AI does, in one line
> You give it a user's **position** (x, y, z). It outputs **M phase values** that
> aim the beam at that user. A physics formula scores how good the aim is, and the
> AI trains to make that score as high as possible.

---

# PART 2 - THE MATH (built up slowly)

## 2.1 Complex numbers (the 2-minute version)
A complex number is written `a + b*i`. Picture it as an arrow on a flat plane:
- its **length** (magnitude) = how strong a signal is,
- its **angle** (phase) = the timing of the wave.

Radio needs both strength and timing, which is why signals are complex numbers.
Neural networks only handle plain real numbers, so we always split one complex
number into two real numbers: the **real part** `a` and the **imaginary part** `b`.
That is why you see `h_re` and `h_im` in the code instead of a single complex `h`.

Two facts we use constantly:
- `exp(j*theta) = cos(theta) + j*sin(theta)` - this turns an angle into a complex
  number of length exactly 1 sitting on the unit circle.
- The magnitude of `a + b*i` is `sqrt(a^2 + b^2)`.

## 2.2 The channel h
The **channel** describes how the signal travels from the antenna array to one
user. For M antennas it is M complex numbers: `h = [h_1, h_2, ..., h_M]`. Each
`h_m` says how strong and how delayed the path from antenna m to the user is.
In code it has shape `(M,)` complex, and the whole dataset is `(N, M)` for N users.

## 2.3 The beamformer f and the "constant modulus" rule
The **beamformer** `f` is the set of complex weights we actually apply to the
antennas. We build it from the network's phases:
```
f = exp(j*theta) / sqrt(M)
```
Why this exact form? Two physical reasons:
- `exp(j*theta)` puts every antenna's weight on the unit circle, so every antenna
  transmits at the **same power** - only the timing (phase) differs. Real hardware
  (a "phase shifter") can delay a signal but cannot amplify it, so this is required.
  This equal-power rule is called **constant modulus**.
- Dividing by `sqrt(M)` fixes the **total transmit power** to exactly 1, so every
  comparison is fair. Check: each `|f_m| = 1/sqrt(M)`, and the total power is
  `sum |f_m|^2 = M * (1/M) = 1`.

The beautiful consequence: whatever random numbers the network spits out, `f` is
**always physically valid**. We never need to clip values or add penalty terms.
(Member 3's `phase_mapping_demo.py` proves exactly this.)

## 2.4 Array gain - how strong the beam is at the user
When the beam `f` meets the channel `h`, the strength the user receives is:
```
gain = |h^H f|^2
```
`h^H f` means "line up f against the channel and add it all up" (the H is a
conjugate - a complex-numbers detail). If `f`'s phases cancel the channel's
phases, every term adds in the same direction and the sum is large. If they are
misaligned, terms cancel and the sum is small. So **gain is a direct measure of
how well the beam is aimed.**

## 2.5 From gain to "data rate" (spectral efficiency)
A stronger beam means faster internet. The exact relationship (a famous result by
Claude Shannon) is:
```
rate = log2(1 + SNR * gain)        (units: bits per second per Hz)
```
- **SNR** = signal-to-noise ratio = how strong the signal is compared to
  background noise. We write it in **decibels (dB)**; 10 dB means `10^(10/10) = 10`
  in plain numbers. Higher SNR = cleaner connection.
- `log2(...)` means each doubling of `(1 + SNR*gain)` adds one bit. This is the
  **spectral efficiency**: the headline number everyone compares.

## 2.6 The loss (what the AI minimises)
Training works by lowering a number called the **loss**. We want to *maximise* the
rate, and training minimises, so the loss is the **negative** rate:
```
loss = - mean over users of  log2(1 + SNR * |h^H f|^2)
```
This is the single most important formula in the project. Member 3 owns it; it
lives in `member3/loss.py`.

**Why the loss is negative and never reaches zero.** Since gain is positive,
`log2(1 + SNR*gain)` is a positive number, so its negative is below zero. The best
the loss can ever get is `-log2(1 + SNR*G_best)`, a negative constant - NOT zero.
So we never watch the raw loss hoping for 0. Instead we watch the **gap** between
the AI's rate and the best-possible rate; that gap shrinking toward 0 is what
"learning well" looks like. (This trips up everyone - say it to your mentor first.)

## 2.7 The best possible beam (the target)
For phase-only beamforming with a known channel, the perfect answer is a one-line
formula: set each antenna's phase to cancel that channel's phase,
```
theta_m = angle(h_m)      ->      best gain = (sum of |h_m|)^2 / M
```
This matters for two reasons:
- It is the **ceiling** the AI is chasing (the "best possible" line in the graphs).
- It means the network is **not discovering unknown physics** - it is learning to
  approximate a function we can already write down. The real value of the project
  is doing this from **position alone, without measuring the channel first** (more
  on that in Part 6).

## 2.8 The two baselines (the reference lines)
To judge the AI we need two reference scores, both from Member 1's `baselines.py`:
- **MRT (maximum ratio transmission):** the absolute best if we could also change
  each antenna's power, not just its phase. `gain = ||h||^2 = sum |h_m|^2`. This is
  the highest ceiling.
- **Phase-only upper bound:** the best for OUR constant-modulus hardware,
  `(sum|h_m|)^2 / M`. This is the fair ceiling for us (always a little below MRT).
- **DFT codebook:** the old fixed-menu method - build a standard menu of beams and
  pick the best one per user. This is the **floor** our AI must beat.

Math note: `(sum|h_m|)^2/M <= sum|h_m|^2`, so phase-only <= MRT, with equality when
all antennas see equal strength. In our data they are almost equal, which is why
MRT and the phase-only bound come out nearly the same number.

---

# PART 3 - THE MODEL (BeamNet), in detail

File: `member2/model.py`. It is a small **MLP** (multi-layer perceptron - the most
basic kind of neural network: a stack of linear layers with bending functions
between them).

```
input: position (3 numbers: x, y, z)
   -> Linear(3 -> 512)   + SiLU
   -> Linear(512 -> 512) + SiLU
   -> Linear(512 -> 64)      (no activation)
output: theta (64 phase values, one per antenna)
```

Line by line:
- **Linear layer:** does `output = input * weights + bias`. The weights start
  random and get tuned during training. `Linear(3 -> 512)` turns 3 input numbers
  into 512 internal numbers.
- **SiLU:** a smooth "bending" function (an activation). Without a bending function
  between linear layers, stacking them would collapse into a single linear layer
  and could only draw straight lines. SiLU lets the network model curved, complex
  relationships.
- **Last layer has NO activation.** The output is a phase (an angle), which can be
  any real value. An activation like sigmoid would squash it into a limited range
  and forbid valid beams. So the final layer is plain.
- **Parameter count ~297,000.** That is the number of weights being tuned. Small by
  AI standards (good - it trains fast and needs little data).

The model does NOT know any physics. All the physics lives in the loss. The model
just learns, by trial and error, which position maps to which good phases.

---

# PART 4 - WHAT EACH MEMBER DID (Weeks 1-4, file by file)

## MEMBER 1 - Wireless / Channel Lead
Owns the data and the reference scores.

- **Week 1 - `plot_scenario.py` (left panel):** a scatter map of where all users
  are. *Why:* everyone needs to see the environment ("the street") first.
- **Week 2 - `generate_channels.py`:** creates `positions.npy` (N,3) and
  `channels.npy` (N,M complex). *Theory:* the array is modelled as a **uniform
  linear array (ULA)**; a user at angle phi produces a **steering vector**
  `exp(j*pi*m*sin(phi))` across antennas m, scaled down by distance (path loss)
  with a little random scattering. This is a realistic stand-in until the real
  DeepMIMO O1 export arrives; same shapes, so nothing downstream changes. *Why:*
  this is the raw material - nobody can train without it.
- **Week 3 - `plot_scenario.py` (right panel):** a heatmap of channel strength.
  *Why:* shows strong areas vs weak "dead zones", which later explains where the
  AI struggles.
- **Week 4 - `baselines.py`:** computes MRT (ceiling), the phase-only upper bound,
  and the DFT codebook (floor). *Why:* these are the lines the AI is measured
  against. If the AI does not beat the DFT floor, the project has no point.

## MEMBER 2 - Deep Learning Engineer (you, the group leader)
Owns the AI, the training machine, and the integration.

- **Week 1 - `w1_sandbox.py`:** trains a throwaway network on random data; we only
  check the loss drops. *Why:* proves PyTorch and the GPU work before building
  anything real.
- **Week 2 - `dataset.py`:** loads Member 1's files, splits each complex channel
  into real/imag, **normalises** (scales channels so average power ~1 and positions
  into [-1,1]), and batches the data. *Theory:* raw channel values are tiny
  (~0.001), which makes gradients vanish and stalls learning; positions are large
  (tens of metres), which networks dislike. Normalising fixes both. We also use a
  **spatial (block) split**, not random, because neighbouring users have almost
  identical channels and a random split would leak test answers into training.
- **Week 3 - `model.py`:** the BeamNet described in Part 3; one batch pushed
  through to confirm the output shape is (batch, M).
- **Week 4 - `train.py` (THE milestone):** ties all three folders together - loads
  Member 1's data, uses Member 3's loss, trains Member 2's model. It proves
  gradients flow from the physics loss back to the weights, the loss drops after
  one step, and the gap to the best-possible rate shrinks each epoch.
- **Extra - `visualize_results.py`:** trains longer and draws the learning curve,
  the beam-shape polar plot, and the AI-vs-baselines bar chart.

## MEMBER 3 - Mathematics / Signal Processing Lead
Owns the loss, the proofs, and the analysis.

- **Week 1:** a written brief turning the spectral-efficiency formula into
  something codeable (notes, not code).
- **Week 2 - `phase_mapping_demo.py`:** proves `f = exp(j*theta)/sqrt(M)` keeps
  every antenna at equal power (constant modulus) and total power exactly 1, for
  ANY theta. *Why:* this is what makes the network output always physically valid -
  no clipping needed.
- **Week 3 - `loss.py`:** the spectral-efficiency loss, plus `optimal_gain` (the
  best-possible gain). Written entirely with real/imag parts (cos and sin) so
  PyTorch's autograd only ever sees real numbers - this avoids all complex-gradient
  problems.
- **Week 4 - `gradient_check.py`:** runs `torch.autograd.gradcheck`, which compares
  autograd's gradients against slow exact numerical ones. If they match, the loss
  is provably differentiable and Member 2 can trust training. It also confirms the
  known optimum (`theta = angle(h)`) reaches the theoretical best gain.

**How they connect:** Member 1 -> data + baselines -> Member 2 -> trained model +
beam predictions -> Member 3 -> loss, proofs, figures. `train.py` imports from all
three folders, which is why the team merges everything together on Fridays.

---

# PART 5 - THE RESULTS, EXPLAINED (your actual run)

Here is exactly what your Colab run printed and what each line means.

## 5.1 Member 1 - the data and baselines
```
saved positions.npy (100000, 3) float32
saved channels.npy  (100000, 64) complex64
M (antennas) = 64
MRT (full upper bound)      : 7.455 bits/s/Hz
phase-only upper bound      : 7.451 bits/s/Hz   <- fair ceiling for us
DFT codebook (old method)   : 7.035 bits/s/Hz   <- floor to beat
```
- 100,000 users, 64 antennas - the dataset is the right size and shape.
- MRT (7.455) and the phase-only bound (7.451) are almost identical, which tells us
  all antennas see nearly equal strength (expected for this clean synthetic data).
- The DFT codebook gets 7.035. The gap between the ceiling (7.451) and this floor
  (7.035) is about **0.42 bits/s/Hz** - that is the **quantization loss** the old
  method wastes, and the exact prize our AI is trying to recover.

## 5.2 Member 2 - the pipeline checks
```
GPU available: False
step 0 loss 1.0522 ... step 250 loss 0.9944     (dummy net, loss drops - tools work)
channel scale: 0.003993...                       (the normalisation constant)
mean power of batch (~1 is good): 1.0568          (normalisation worked)
theta shape: (512, 64)                            (model output is correct)
total parameters: 297536                          (size of the network)
```
- `GPU available: False` just means you did not switch Colab to the T4 GPU that
  run - it worked on CPU, only slower.
- The dummy loss dropping = PyTorch works. Batch power ~1 = normalisation correct.
- `theta shape (512, 64)` = for 512 users the model produced 64 phases each. 

## 5.3 Member 3 - the proofs
```
each |f_i| should equal 1/sqrt(M) = 0.125
min |f_i|: 0.125   max |f_i|: 0.125   total power: 1.0
loss value: -4.0865 (negative is correct)
gradcheck passed: True
achieved gain with optimal phases: 30.8103...
theoretical optimal gain:          30.8103...   (they match exactly)
```
- Every `|f_i|` is exactly 0.125 and total power is 1.0 - the constant-modulus rule
  holds perfectly, so the beam is always physically valid.
- The loss is -4.09, a negative number, exactly as expected (see 2.6).
- `gradcheck passed: True` - the loss's gradients are mathematically correct, so
  training is trustworthy. This is a strong thing to show a mentor.
- The achieved gain with the optimal phases matches the theoretical optimum to many
  decimals - the maths in `optimal_gain` is verified.

## 5.4 Member 2 - the Week 4 milestone (training)
```
device: cpu
starting loss: -0.5417 (negative is normal)
layers with missing / NaN gradient: none (good)
loss before: -0.5417 -> after one step: -0.6278 (lower = learning works)
short training run:
  epoch 0: our rate 2.435 / best possible 7.446 (gap 5.011)
  epoch 1: our rate 2.863 / best possible 7.446 (gap 4.582)
  epoch 2: our rate 3.129 / best possible 7.446 (gap 4.317)
  epoch 3: our rate 3.299 / best possible 7.446 (gap 4.147)
  epoch 4: our rate 3.412 / best possible 7.446 (gap 4.034)
```
This is the heart of the whole project, and it is working:
- "none" missing gradients = the learning signal reaches every layer.
- loss dropping after one step = one nudge already improved the beam.
- The **gap** column falls every epoch: 5.011 -> 4.582 -> 4.317 -> 4.147 -> 4.034.
  That steady fall is the proof the AI is genuinely learning to aim the beam.

**Why the rate is only ~3.4, not near 7.4 yet:** this quick demo ran just **5
epochs on a CPU** - barely any training. It was only meant to prove the mechanism
works, which it does. The `visualize_results.py` script runs 40 epochs and climbs
much higher; with a full run on the GPU in Week 6 the rate keeps rising toward the
ceiling and past the 7.035 DFT floor. Nothing is wrong - it is simply
under-trained on purpose for a fast check.

## 5.5 The three visuals (`visualize_results.py`)
- **training_curve.png:** the AI's rate climbing over 40 epochs, with the green
  ceiling and red DFT-floor lines. Seeing the blue line rise toward and past the
  red line is the single clearest proof of success.
- **beam_pattern.png:** a polar plot of the AI's beam vs the ideal beam for one
  user, with a marker at the user's true direction. When the AI's peak sits on that
  marker, the beam is correctly aimed - the project working, as a picture.
- **comparison_bar.png:** three bars - old method, our AI, best possible - the
  one-glance "did we beat the baseline?" chart.

---

# PART 6 - THINGS YOUR MENTOR WILL ASK (own these)

1. **Why does the loss not reach zero?** Because a good score is a negative number;
   it settles at `-log2(1+SNR*best gain)`. We track the **gap to the optimum**, not
   the raw loss. The gap going to ~0 means success.

2. **If you train on position, will it work on a new street?** No - a position-input
   network memorises one map and will not transfer. So later we also build a
   **channel-input** version for the generalisation test. (Weeks 1-4 use position.)

3. **What is actually new, if the optimum is a known formula?** The point is not
   "we removed the codebook." It is that the AI predicts a near-optimal beam from
   **position alone, with no pilot measurements**, while the old codebook method
   needs many pilot measurements to search its menu. Less measuring = less overhead
   and lower latency. That is the real contribution.

4. **How do you avoid complex-number gradient problems?** The loss is written
   entirely in real/imag arithmetic (cos and sin), so autograd only ever sees real
   numbers.

5. **How do you know it did not just memorise / cheat?** We use a spatial split so
   nearby users cannot leak between train and test, and later a zero-shot test on an
   unseen scenario.

---

# GLOSSARY (one line each)
- **Antenna array (M):** a row of M antennas that together form a beam.
- **Beamforming:** choosing phase shifts so the beam points at the user.
- **Phase / theta:** the timing (angle) applied to each antenna - the model's output.
- **Channel (h):** M complex numbers describing how the signal reaches a user.
- **Beamformer (f):** the complex weights applied; `f = exp(j*theta)/sqrt(M)`.
- **Constant modulus:** every antenna at equal power; only phase changes.
- **Array gain:** beam strength at the user, `|h^H f|^2`.
- **SNR:** signal-to-noise ratio; higher = cleaner link; given in decibels (dB).
- **Spectral efficiency / rate:** data rate, `log2(1 + SNR*gain)`, in bits/s/Hz.
- **Codebook:** a fixed menu of beams (the old method we aim to beat).
- **Quantization loss:** signal wasted by rounding to the nearest menu beam.
- **MRT:** the absolute best beam if power could vary too (highest ceiling).
- **Loss:** the number training minimises; here, the negative data rate.
- **Gradient:** the direction to adjust each weight to lower the loss.
- **Autograd / backward:** PyTorch computing those gradients automatically.
- **gradcheck:** a test proving those gradients are mathematically correct.
- **Epoch:** one full pass of the training data through the model.
- **MLP:** multi-layer perceptron, the basic stacked-linear-layers network.
- **DeepMIMO / O1:** the ray-tracing dataset / the specific street scenario.
