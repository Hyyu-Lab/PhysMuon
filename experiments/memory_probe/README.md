# Read-only probe of the optimal memory

The measurements behind the supplementary note "Read-only probe: the optimal memory grows during training". The probe
runs next to an unmodified training run of the leaderboard record configuration (single-mode momentum) and only reads
gradients and optimizer state: from two half-batch gradients it estimates, along the singular directions of the
momentum, the strength of the drift D and of the batch noise T, and reports K* = sqrt(D / T) for five groups of
directions in each of the 72 hidden matrices (360 measuring units). The optimal memory is n* = (sqrt(1 + 4 / K*²) − 1) / 2.

| | |
|---|---|
| [`data/buckets_s<seed>.csv`](data) | eight runs (seeds 0–7): for every step, ln K* of the five groups (median over the 72 matrices) |
| [`analyze.py`](analyze.py) | recomputes the numbers of the note; its output is [`expected_output.txt`](expected_output.txt) |

```bash
python analyze.py                               # the two windows of seed 0, the 40 late/early ratios of the eight runs
python analyze.py --raw slim_s0.jsonl.gz        # also the spread and the ordering over the 360 units of seed 0
```

The series of all 360 units at every step (`slim_s<seed>.jsonl.gz`, 79 MB for the eight runs) are public in
[bimaxwell-track3-reproduction](https://github.com/orange4664/bimaxwell-track3-reproduction) (folder `kstar_probe/`),
together with the per-seed logs of the record configuration; the files in `data/` are identical to the
`buckets_s<seed>.csv` there.
