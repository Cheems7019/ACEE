# IHDP experiment

This directory contains the ACEE evaluation on the semi-synthetic Infant Health and Development Program (IHDP) benchmark used in the manuscript.

## Included data

`data/ihdp_npci_1-100.train.npz` and `data/ihdp_npci_1-100.test.npz` contain the 100 retained benchmark realizations used by `IHDP_ACEE.py`. They are intentionally tracked in the repository; generated estimates and summaries under `IHDP/results/` are ignored.

## Run

From the repository root:

```bash
python IHDP/IHDP_ACEE.py --device cuda:0
python IHDP/summary_acee_ihdp.py
```

Use `--rep_start` and `--rep_end` to run only a subset of realizations. Run either script with `--help` for all options. The summary script accepts optional comparison-result CSVs, but competitor implementations are not distributed here.
