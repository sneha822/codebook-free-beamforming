# The Road to the Finish (Weeks 5-11) - all three members

Where we are: Weeks 1-4 are done. We have data, a working model, a verified loss,
and proof the AI learns (the gap shrinks). From here, each week turns that working
pipeline into real results, then into a finished, presentable project.

**The finish line (what "done" looks like in Week 11):**
- A trained "golden model" that beats the DFT codebook baseline.
- One headline graph: Capacity vs SNR, showing AI between the MRT ceiling and DFT floor.
- Proof it generalises to an unseen scenario (not just memorising).
- A clean GitHub repo + a CLI where you type a position and get a beam.
- A short paper/report with the math, the figures, and the results.

Each week below lists the goal in plain words, then what **each member** does, then
the shared deliverable to show the mentor.

---

## WEEK 5 - Prove it can truly learn (overfit 10 samples)
**Goal:** before training on everything, prove the model can perfectly learn just
10 users. If it can memorise 10, the architecture and loss are sound.

- **Member 1:** provide the DFT codebook number as the "floor" to mark on the plot.
- **Member 2 (you):** write the overfit script - 10 fixed users, no shuffle, ~2000
  steps; tune the learning rate until the gap to the optimum is almost 0.
- **Member 3:** confirm the loss behaves (the gap, not the loss, goes to ~0) and
  check the maths of the optimum on these 10 users.

**Deliverable:** a curve showing the gap collapse to near zero on the 10 samples.
**Why it matters:** this is the classic sanity check - if a model can't overfit a
tiny set, something is broken. Passing it means we can trust full training.

---

## WEEK 6 - First real training run + see the beams
**Goal:** train on the full 100,000 users for 50-100 epochs and look at the beams.

- **Member 1:** provide the MRT ceiling line; help read which users are hard (dead
  zones).
- **Member 2:** finish `train.py` (epoch loop, validation each epoch, save the best
  model); run the full training; if the model underfits, add **Fourier features**
  (a trick that helps a network learn the sharp position->beam relationship).
- **Member 3:** make the **polar beam plots** comparing the AI beam vs the ideal
  beam for a few users.

**Deliverable:** training/validation curves + polar beam plots (AI vs ideal).
**Why it matters:** first evidence the AI works on the real scale, not just a toy.

---

## WEEK 7 - Scale the array (M = 16, 32, 64) + failure analysis
**Goal:** show the method works for different array sizes, and understand where it
fails.

- **Member 1:** the theory graph of how capacity grows with more antennas (M).
- **Member 2:** train three models (M = 16, 32, 64) with the same settings; make a
  table of parameters, memory, training time, and final rate; export the worst 5%
  of users.
- **Member 3:** the **failure analysis** - take those worst users and explain why
  (extreme blocked paths, grid edges, very weak signal).

**Deliverable:** a three-size comparison table + a short failure report.
**Why it matters:** shows the method is general, and being honest about failures is
exactly what makes research credible.

---

## WEEK 8 - The headline benchmark (Capacity vs SNR)
**Goal:** the single most important figure of the whole project.

- **Member 1:** compute the MRT and DFT baseline curves across SNR from -10 to 20 dB.
- **Member 2:** write `evaluate.py` to produce the AI curve across the same SNR
  range; also add quantized versions (3-bit, 5-bit) to show what rounding costs.
- **Member 3:** compute "quantization loss recovered = (AI - DFT) / (MRT - DFT)"
  and write the explanation.

**Deliverable:** one polished graph overlaying AI, MRT (ceiling), DFT (floor)
across SNR - plus the single sentence "we recovered X% of the codebook's loss."
**Why it matters:** this graph IS the result of the project.

---

## WEEK 9 - Speed and user tracking
**Goal:** show the AI is fast and tracks a moving user smoothly.

- **Member 1:** provide a user trajectory (a path walking through the street).
- **Member 2:** benchmark inference time (AI vs the DFT exhaustive search), with
  proper warm-up and timing; produce a latency table.
- **Member 3:** plot the phase continuity along the trajectory (the beam should
  follow the user smoothly, not jump).

**Deliverable:** a latency table + a smooth beam-tracking plot.
**Why it matters (and the honest framing):** in pure software the DFT search is also
a fast matrix multiply, so raw compute time is close. Our real advantage is **no
pilot measurements needed** - the AI aims from position alone, while the codebook
needs many over-the-air measurements. Report both.

---

## WEEK 10 - The big test: works on an unseen scenario?
**Goal:** prove the model learned real physics, not just memorised the O1 street.

- **Member 1:** source a different, unseen scenario (a new street / dataset) with
  the same array size.
- **Member 2:** freeze the "golden model" (save weights + config + normalisation,
  tag it in git); run it on the unseen scenario with no further tuning; if zero-shot
  is weak, also show a few-shot fine-tune curve (still a valid result).
- **Member 3:** compare AI vs baselines on the new scenario and help phrase the
  claim carefully.

**Deliverable:** a table of AI vs baselines on the unseen street.
**Why it matters:** this is the difference between "a cool demo" and "it learned
wireless physics." (Remember from Part 6: a position-input model may not transfer -
this is where the channel-input version earns its place.)

---

## WEEK 11 - Package it and present
**Goal:** turn everything into a finished, reproducible project.

- **Member 1:** final clean datasets + documentation of how they were made.
- **Member 2:** build the CLI (`predict.py`: type x, y, z -> get the beam); pin
  `requirements.txt`; write the README; reproduce the golden result on a clean
  machine from scratch.
- **Member 3:** compile all figures and the full mathematical derivations into the
  report/paper.

**Deliverable:** the finished GitHub repo + a live CLI demo + all figures and math.
**Why it matters:** this is what the mentor grades and what goes on your resume.

---

# HOW WE ACTUALLY MAKE IT TO THE END

## The three "do not skip" milestones
1. **Week 5 overfit** - if this fails, stop and fix before scaling. (Done right, it
   guarantees the rest can work.)
2. **Week 8 Capacity-vs-SNR graph** - this is the result. Everything before it
   builds to it; everything after it supports it.
3. **Week 10 unseen-scenario test** - this is what proves it is real research.

## Who owns what, in one line each
- **Member 1 (Wireless):** always one step ahead on data and baselines - every week
  needs their numbers to compare against.
- **Member 2 (you, leader):** the engine - training, experiments, the CLI, and
  keeping the GitHub repo merged and runnable.
- **Member 3 (Maths):** the proofs and every figure - turns raw outputs into the
  graphs and explanations the mentor sees.

## The weekly rhythm that keeps us on track
- Mon-Thu: each member does their 1-hour solo task.
- Friday (2 h together): merge everyone's code into the one GitHub repo, run it end
  to end, and build that week's single deliverable. Never debug live on Sunday.
- Sunday: show the mentor one working thing, state blockers, get guidance.
(Full detail in `FRIDAY_SESSIONS.md`.)

## Risks to watch (and the plan if they happen)
- **Real DeepMIMO data is late** -> keep using the synthetic stand-in; the shapes
  match, so swapping it in changes nothing else.
- **Model underfits in Week 6** -> add Fourier features + a wider/deeper network.
- **Zero-shot fails in Week 10** -> fall back to the channel-input model and/or a
  few-shot fine-tune curve (an honest, publishable result either way).
- **No GPU** -> train on Colab's free T4; keep datasets small enough to fit.
- **A week slips** -> Weeks 7 and 9 are the most compressible; protect Weeks 5, 8, 10.

## What we can already say is working (as of Week 4)
Data pipeline, model, verified loss, and proof of learning (the shrinking gap).
The foundation is solid - the remaining weeks are about results and polish, not
rebuilding anything.
