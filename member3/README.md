# Member 3 - Mathematics / Signal Processing (Weeks 1-4)

My job: the scoring rule (loss), the proofs, and the comparison analysis.

## Files
- `loss.py` - the spectral-efficiency loss + the optimal (best-possible) gain.
- `phase_mapping_demo.py` - proves real outputs map to valid constant-modulus phases.
- `gradient_check.py` - formally verifies the loss is correctly differentiable.

## Run order
```bash
pip install torch
python phase_mapping_demo.py     # Week 2
python loss.py                   # Week 3
python gradient_check.py         # Week 4
```

## What each week delivers and why
- **Week 1:** a one-page brief turning the spectral-efficiency formula
  `L = -log2(1 + SNR*|h^H f|^2)` into something codeable (notes, not code).
- **Week 2:** `phase_mapping_demo.py` shows that `f = exp(j*theta)/sqrt(M)` keeps
  every antenna at equal power for ANY theta - so the network output is always
  physically valid, no clipping needed.
- **Week 3:** `loss.py` computes the loss on dummy tensors with no autograd error.
  It's written in real/imag parts so there are no complex-gradient problems.
- **Week 4:** `gradient_check.py` passes `torch.autograd.gradcheck`, which proves
  the gradients are mathematically correct - so Member 2 can trust training.
  It also confirms the known optimum (theta = angle(h)) reaches the theoretical
  best gain, which is the target the AI aims for.

Key point for the mentor: the loss never reaches zero (a good score is a negative
number). What we actually track is the GAP between the AI's rate and this optimal
rate - that gap going to ~0 is what "learning well" means.
