# Member 3 - Mathematics
# Week 2 task: prove that the network's plain real-number outputs can be turned
# into valid complex phases WITHOUT breaking the physical rule that every
# antenna sends at the same power (constant modulus).
#
# The rule: f = exp(j*theta)/sqrt(M). We check that |f_i| is the same for every
# antenna no matter what theta is, and that the total power is exactly 1.

import torch

M = 64

# pretend these came out of the network - any real numbers at all
theta = torch.randn(M) * 10

# map to the complex beamformer
f = torch.exp(1j * theta) / (M ** 0.5)

magnitudes = f.abs()
print("each |f_i| should equal 1/sqrt(M) =", round(1 / M ** 0.5, 5))
print("min |f_i|:", round(magnitudes.min().item(), 5))
print("max |f_i|:", round(magnitudes.max().item(), 5))
print("total power (should be 1.0):", round((magnitudes ** 2).sum().item(), 5))
print("\nConclusion: any real theta gives a valid constant-modulus beamformer,")
print("so the network output never needs clipping or extra penalty terms.")
