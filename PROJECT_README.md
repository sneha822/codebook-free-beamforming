# Codebook-Free Continuous Phase-Shift Predictor

A neural network that aims a mmWave antenna beam by predicting continuous phase
values directly, instead of picking from a fixed codebook. Trained against a
physics-based spectral-efficiency loss.

## Folder layout (one folder per member)
```
member1/   Wireless  - data + baselines
member2/   DL engineer - data pipeline, model, training
member3/   Maths     - loss function, proofs, gradient checks
```

## How the pieces connect
```
member1 (data, baselines) --> member2 (model + training) --> member3 (loss, proofs)
             \__________ baselines + analysis feed the final results ________/
```
`member2/train.py` is the glue: it imports the data from `member1` and the loss
from `member3`, so running it exercises all three parts together.

## How to run everything (Weeks 1-4)
Easiest on Google Colab (free GPU). In a notebook:
```python
!pip install torch numpy matplotlib
```
Then run, in order:
```bash
# 1. make the data (Member 1)
cd member1 && python generate_channels.py && cd ..
# 2. check the pipeline (Member 2)
cd member2 && python w1_sandbox.py && python dataset.py && python model.py && cd ..
# 3. check the maths (Member 3)
cd member3 && python phase_mapping_demo.py && python loss.py && python gradient_check.py && cd ..
# 4. the Week 4 milestone: full integration (Member 2)
cd member2 && python train.py
```

## Where things live (important)
- Code: here / GitHub.
- Big data + trained models: Google Drive, NOT this OneDrive folder (OneDrive can
  corrupt large files while syncing). The synthetic data here is small, so it's
  fine for now.

See each folder's README for the per-week explanation, and `PROJECT_BRIEF.md` for
the full project write-up.
