# Member 1 - Wireless
# Week 2 task: create the channel dataset the others will train on.
#
# We don't have the real DeepMIMO export yet, so this makes realistic
# stand-in data with the SAME shapes, so Member 2 and Member 3 are not blocked.
# When the real .npy files are ready, just replace this file's output.
#
# Physics idea: the tower is a uniform linear array (ULA) of M antennas at the
# origin. Each user sits at some (x, y, z). The signal reaches the array at an
# angle, so the channel is a "steering vector" (a phase ramp across antennas)
# times a distance-based strength. That is what real mmWave channels look like.

import numpy as np

def make_dataset(n_users=100000, M=64, seed=0):
    rng = np.random.default_rng(seed)

    # scatter users over a street-like area (meters)
    x = rng.uniform(-50, 50, n_users)
    y = rng.uniform(10, 110, n_users)
    z = rng.uniform(1.0, 2.0, n_users)       # roughly phone height
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    # angle of the user as seen by the array
    phi = np.arctan2(x, y)

    # steering vector: antenna m gets an extra phase of pi * m * sin(phi)
    m = np.arange(M)
    steering = np.exp(1j * np.pi * np.outer(np.sin(phi), m))   # (n_users, M)

    # signal gets weaker with distance (path loss)
    dist = np.sqrt(x**2 + y**2 + z**2)
    strength = (1.0 / dist**1.5).reshape(-1, 1)

    # a bit of random scattering so it is not perfectly clean
    noise = 0.05 * (rng.standard_normal((n_users, M)) + 1j * rng.standard_normal((n_users, M)))

    channels = (strength * steering + strength * noise).astype(np.complex64)
    return positions, channels


if __name__ == "__main__":
    pos, h = make_dataset()
    np.save("positions.npy", pos)
    np.save("channels.npy", h)
    print("saved positions.npy", pos.shape, pos.dtype)
    print("saved channels.npy ", h.shape, h.dtype)
    print("M (antennas) =", h.shape[1])
    print("example channel magnitude range:",
          round(float(np.abs(h).min()), 6), "to", round(float(np.abs(h).max()), 6))
