# Member 1 - Wireless
# Week 1 + Week 3 task: show the scenario visually.
#   Week 1 -> a map of where the users are (the "street").
#   Week 3 -> a heatmap of channel strength, so we can see strong vs dead zones.
#
# Run generate_channels.py first so the .npy files exist.

import numpy as np
import matplotlib.pyplot as plt

pos = np.load("positions.npy")
h = np.load("channels.npy")

# strength per user = total power across the antennas
power_db = 10 * np.log10((np.abs(h) ** 2).sum(axis=1) + 1e-12)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

# left: user map
ax[0].scatter(pos[:, 0], pos[:, 1], s=2, alpha=0.3)
ax[0].set_title("User positions (the O1-style street)")
ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")

# right: channel strength heatmap
sc = ax[1].scatter(pos[:, 0], pos[:, 1], c=power_db, s=3, cmap="viridis")
ax[1].set_title("Channel strength (dB) - dark = dead zone")
ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("y (m)")
fig.colorbar(sc, ax=ax[1], label="received power (dB)")

plt.tight_layout()
plt.savefig("scenario.png", dpi=150)
print("saved scenario.png")
