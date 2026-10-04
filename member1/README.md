# Member 1 - Wireless / Channel Lead (Weeks 1-4)

My job: provide the data and the reference scores the AI is measured against.

## Files
- `generate_channels.py` - creates the dataset (`positions.npy`, `channels.npy`).
- `plot_scenario.py` - the user map + channel-strength heatmap.
- `baselines.py` - MRT (ceiling), phase-only upper bound, DFT codebook (floor).

## Run order
```bash
pip install numpy matplotlib
python generate_channels.py     # makes positions.npy + channels.npy
python plot_scenario.py         # makes scenario.png
python baselines.py             # prints the baseline data rates
```

## What each week delivers and why
- **Week 1:** the user map (`scenario.png` left). Shows the "street" we work on.
- **Week 2:** `positions.npy` (N,3) and `channels.npy` (N,M complex). This is the
  data Member 2 loads. Shapes are the contract everyone agrees on.
- **Week 3:** the strength heatmap (`scenario.png` right). Shows strong areas vs
  dead zones, which later explains where the AI struggles.
- **Week 4:** the baselines. MRT is the best-possible score (we can't beat it);
  the DFT codebook is the old method (we must beat it). Our AI's score will sit
  between these two lines.

Note: the data here is a realistic stand-in until the real DeepMIMO O1 export is
ready. Same shapes, so nothing downstream changes when we swap it in.
