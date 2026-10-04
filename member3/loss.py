# Member 3 - Mathematics / Signal Processing
# Week 3 task: the loss function - the scoring rule the whole project trains on.
#
# The network outputs phases theta. The actual beamformer is
#     f = exp(j*theta) / sqrt(M)
# and the gain a user gets is |h^H f|^2.
#
# We write everything using the real and imaginary parts separately (cos and
# sin), so PyTorch's autograd only ever sees real numbers. This avoids all the
# tricky bits of differentiating complex numbers.

import torch


def array_gain(h_re, h_im, theta):
    # gain = |h^H f|^2  with  f = exp(j*theta)/sqrt(M)
    M = theta.shape[-1]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # real and imaginary parts of h^H f
    re = (h_re * cos_t + h_im * sin_t).sum(-1) / M ** 0.5
    im = (h_re * sin_t - h_im * cos_t).sum(-1) / M ** 0.5
    return re ** 2 + im ** 2          # shape (batch,)


def rate_from_gain(gain, snr_db=10.0):
    # spectral efficiency = data rate = log2(1 + SNR * gain)
    snr = 10 ** (snr_db / 10)
    return torch.log2(1 + snr * gain)


def neg_rate_loss(h_re, h_im, theta, snr_db=10.0):
    # we want to MAXIMISE the rate, so we minimise its negative
    gain = array_gain(h_re, h_im, theta)
    return -rate_from_gain(gain, snr_db).mean()


def optimal_gain(h_re, h_im):
    # The best a phase-only beam can do: set each phase to cancel the channel's
    # phase (theta_i = angle(h_i)). Then gain = (sum |h_i|)^2 / M.
    # This is the target the network is trying to reach.
    mag = torch.sqrt(h_re ** 2 + h_im ** 2)
    return mag.sum(-1) ** 2 / h_re.shape[-1]


if __name__ == "__main__":
    # Week 3 proof: the loss runs on dummy tensors with no autograd error.
    torch.manual_seed(0)
    B, M = 8, 64
    h_re, h_im = torch.randn(B, M), torch.randn(B, M)
    theta = torch.randn(B, M, requires_grad=True)

    loss = neg_rate_loss(h_re, h_im, theta, snr_db=10)
    loss.backward()
    print("loss value:", round(loss.item(), 4), "(negative is correct)")
    print("gradient flows into theta:", theta.grad is not None)

    # sanity: the best-case gain should match a brute-force check
    best = optimal_gain(h_re, h_im)
    print("optimal gain of first user:", round(best[0].item(), 4))
