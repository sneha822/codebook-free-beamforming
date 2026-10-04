# Member 1 - Wireless
# Week 4 task: the two reference scores (baselines) the AI is compared against.
#
#   MRT  -> the best score possible if we had a perfect, unconstrained beam.
#           This is the CEILING. The AI can only approach it, never beat it.
#   DFT codebook -> the old "pick from a fixed menu" method.
#           This is the FLOOR. Our AI must beat this to be useful.
#
# Everything here is plain numpy because these are for evaluation, not training.

import numpy as np


def mrt_gain(h):
    # Maximum ratio transmission: point the beam exactly along the channel.
    # Unconstrained beam -> gain = ||h||^2 (sum of per-antenna power).
    return (np.abs(h) ** 2).sum(axis=1)


def phase_only_upper_bound(h):
    # Our hardware can only change phase, not amplitude (constant modulus).
    # Best phase-only beam sets each phase to cancel the channel's phase.
    # gain = (sum of |h_i|)^2 / M. This is the fair ceiling for OUR system.
    M = h.shape[1]
    return (np.abs(h).sum(axis=1) ** 2) / M


def dft_codebook_gain(h, n_beams=None):
    # Build a DFT codebook (the standard fixed menu of beams) and, for each
    # user, pick the beam that gives the most gain - exactly what 5G does.
    M = h.shape[1]
    K = n_beams or M
    m = np.arange(M)
    k = np.arange(K)
    # column k of the codebook is one beam direction
    codebook = np.exp(-2j * np.pi * np.outer(m, k) / K) / np.sqrt(M)   # (M, K)

    # gain for every user against every beam, then take the best beam
    corr = np.abs(h @ codebook) ** 2        # (n_users, K)
    return corr.max(axis=1)                 # best beam per user


def capacity(gain, snr_db=10.0):
    # spectral efficiency (data rate) from a gain value
    snr = 10 ** (snr_db / 10)
    return np.log2(1 + snr * gain)


if __name__ == "__main__":
    h = np.load("channels.npy")
    # normalise the same way training will, so numbers are comparable
    scale = np.sqrt((np.abs(h) ** 2).mean())
    h = h / scale

    mrt = capacity(mrt_gain(h)).mean()
    ub = capacity(phase_only_upper_bound(h)).mean()
    dft = capacity(dft_codebook_gain(h)).mean()
    print(f"MRT (full upper bound)      : {mrt:.3f} bits/s/Hz")
    print(f"phase-only upper bound      : {ub:.3f} bits/s/Hz  <- fair ceiling for us")
    print(f"DFT codebook (old method)   : {dft:.3f} bits/s/Hz  <- floor to beat")
