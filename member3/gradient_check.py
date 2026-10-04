# Member 3 - Mathematics
# Week 4 task: formally verify that the loss is correctly differentiable, using
# PyTorch's built-in gradcheck. This compares autograd's gradients against
# slow-but-exact numerical gradients. If they match, the maths is sound and
# Member 2 can trust loss.backward() in training.
#
# gradcheck needs float64 (double precision) to be accurate.

import torch
from loss import neg_rate_loss, array_gain, optimal_gain, rate_from_gain

torch.manual_seed(0)
B, M = 4, 16

h_re = torch.randn(B, M, dtype=torch.float64)
h_im = torch.randn(B, M, dtype=torch.float64)
theta = torch.randn(B, M, dtype=torch.float64, requires_grad=True)

# wrap so gradcheck varies only theta
def fn(t):
    return neg_rate_loss(h_re, h_im, t, snr_db=10)

ok = torch.autograd.gradcheck(fn, (theta,), eps=1e-6, atol=1e-4)
print("gradcheck passed:", ok)

# also confirm the network can at best reach the optimal gain, not exceed it
theta_opt = torch.atan2(h_im, h_re)          # the closed-form best phases
g_model = array_gain(h_re, h_im, theta_opt)
g_best = optimal_gain(h_re, h_im)
print("achieved gain with optimal phases:", g_model[0].item())
print("theoretical optimal gain:        ", g_best[0].item())
print("they should match ->", torch.allclose(g_model, g_best, atol=1e-6).item())
