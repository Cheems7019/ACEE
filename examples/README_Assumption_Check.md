# Proxy-assumption diagnostic

This example implements the diagnostic study for the proxy-variable condition discussed in the manuscript. A rank-two SVD representation supplies proxy factors, a conditional diffusion model estimates the required conditional law, and an MCODEC statistic tests the resulting conditional independence.

## Scenarios

For `i = 1, ..., p`, both scenarios begin with

```text
X_i = H^2 + epsilon_i.
```

The second block differs:

```text
Scenario 1: X_{p+i} = H(X_i - H^2) + H + epsilon_{p+i}
Scenario 2: X_{p+i} = X_i - H^2 + H + epsilon_{p+i}
```

Scenario 1 violates the proxy condition studied in the paper; Scenario 2 satisfies it. In the script’s variable aliases, the diagnostic tests whether the second-block residual is independent of its proxy component conditional on the corresponding first-block observation and proxy component.

## Run

From the repository root:

```bash
python examples/Assumption_Check.py
```

The default output directory is `examples/`, regardless of the current working directory. A representative custom run is:

```bash
python examples/Assumption_Check.py \
  --n 500 --p 20 --seed 1 --n_epochs 3000 \
  --D 500 --D_batch 250 --device cuda:0
```

Use `--device cpu` when CUDA is unavailable. Run `python examples/Assumption_Check.py --help` for all settings.

## Interpretation

- `p-value <= 0.05`: reject the conditional-independence null; evidence against the proxy condition.
- `p-value > 0.05`: fail to reject; the diagnostic does not find evidence against the condition.

The significance threshold used for this final interpretation is distinct from the `--alpha` training-validation setting.

## Outputs

```text
examples/
├── data/Assumption_Check_sc1/
├── data/Assumption_Check_sc2/
├── ckpt/Assumption_Check_sc1/
├── ckpt/Assumption_Check_sc2/
├── results/Assumption_Check_sc1/
├── results/Assumption_Check_sc2/
└── results/Assumption_Check/
    ├── config.json
    └── singular_values.csv
```

Open `Assumption_Check.ipynb` after the script completes to visualize the score distributions and compare the two scenarios. Generated data, checkpoints, and results are ignored by Git.
