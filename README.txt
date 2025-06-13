README.txt
==========

This code is licensed for non-commercial research and demonstration only. For commercial licensing, see LICENSE.txt or contact [contact@fieldandstring.com].

---

Jocko says: “Few things actually matter. Align your energy and your will on the things that do.”

This repo is a bare-bones, fully visible demonstration of modular fieldblock pruning for neural networks.
Everything you need to run, log, plot, and compare blockwise pruning is here—nothing hidden, nothing implied.

---

WHAT THIS IS
------------

- A toymachine for blockwise pruning: freeze blocks by activity, revive if loss spikes, log and plot everything.
- All dials up front: you control how many blocks, how aggressive pruning is, and how often it happens.
- Plots every relevant metric, including block sleep/wake history (“field heatmap”), loss, entropy, effort, and FLOPS.
- Example logs show both “PruneML” (pruning on) and Adam (baseline) modes, including wall time.
- No proprietary math, kernels, or IP exposed.
- This is demo- and patent-safe—use it for research, talks, or hardware benchmarking.

---

HOW TO RUN
----------

1. **Install dependencies**
   (Python 3.8+, PyTorch, torchvision, matplotlib, numpy)
   pip install torch torchvision matplotlib numpy

2. **Basic usage**
   python pruniverse_toymachine.py

3. **Tune all main parameters at the command line**
   python pruniverse_toymachine.py --epochs 10 --num_blocks 16 --prune_every 50 --freeze_n 2 --freeze_threshold 30

4. **Compare to Adam baseline**
   python pruniverse_toymachine.py --adam

5. **All results, logs, and plots are auto-generated.**
   - The main log is pruniverse_log.csv
   - Example logs are in CONSOLE_OUTPUT.txt (see below)

---

KEY COMMAND-LINE ARGUMENTS
--------------------------

--epochs             Training epochs (default: 10)
--batch              Batch size (default: 64)
--field_dim          Total field width (default: 128)
--num_blocks         Number of blocks (default: 16)
--prune_every        Steps between pruning events (default: 50)
--freeze_n           How many blocks to freeze each time (default: 2)
--freeze_threshold   Activity level below which blocks are frozen (default: 30)
--revive_margin      Multiplier for loss spike that revives all blocks (default: 2.0)
--adam               Disables pruning; runs standard Adam baseline

See all options with:
python pruniverse_toymachine.py --help

---

WHAT TO EXPECT
--------------

- **Fieldblock pruning:**
  Blocks with low activity are frozen (disabled); can be revived if loss spikes.
- **Plots:**
  Loss, entropy, effort, FLOPS, and a per-block “sleep/wake” heatmap.
- **Console log:**
  Shows block events, loss spikes, and all settings.
- **Wall time:**
  See direct comparison between pruning and baseline runs.

---

EXAMPLE LOGS
------------

See CONSOLE_OUTPUT.txt for copy-pasteable real runs, including final wall time.

---

FIELD HEATMAP
-------------

After every run, you’ll see a plot showing which blocks slept (dark) or were awake (light) at each training step.

---

WHAT THIS ISN’T
---------------

- No new algorithm, no claims of SOTA.
- Not for direct research use—this is an explainer, a lab notebook, a public artifact.
- No secret fieldblock or kernel math is exposed.

---

CREDITS
-------

Written for engineers, hardware hackers, the patent office, and anyone who needs to see before they believe.

Questions: [contact@fieldandstring.com]

---
