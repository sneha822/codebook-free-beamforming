# Member 2 - DL Engineer (Weeks 1-4)

My job: build the AI and the training machinery, and tie everyone's code together.

## Files
- `w1_sandbox.py` - GPU check + dummy network (proves the tools work).
- `dataset.py` - loads Member 1's data, normalises it, makes batches.
- `model.py` - BeamNet: position in, M phase values out.
- `train.py` - the training loop; uses Member 1's data + Member 3's loss.

## Run order
```bash
pip install torch numpy
python w1_sandbox.py                 # Week 1
python dataset.py                    # Week 2 (needs ../member1/*.npy)
python model.py                      # Week 3
python train.py                      # Week 4 (needs ../member1 and ../member3)
```

## What each week delivers and why
- **Week 1:** a dropping loss on random data -> the PyTorch + GPU setup works.
- **Week 2:** a DataLoader printing correct batch shapes -> data flows in cleanly.
  We split complex channels into real/imag (nets can't eat complex numbers) and
  normalise (raw values are too tiny/large to learn from).
- **Week 3:** BeamNet does a forward pass with the right output shape (batch, M).
  The last layer has no activation because a phase can be any value.
- **Week 4 (the milestone):** `train.py` proves gradients flow from Member 3's
  physics loss back to the weights, and the loss drops after one step. Then a
  short run shows the gap to the best-possible rate shrinking each epoch.

`train.py` is the integration point - it imports from `../member1` and
`../member3`. That is why we merge all folders together on Fridays.
