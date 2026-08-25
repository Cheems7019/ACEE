# Causal-discovery example

This example reproduces the three-variable causal-discovery illustration in the manuscript.

## Data-generating graph

The generated data follow graph G2:

```text
X1 ──▶ X2 ──▶ X3
 └──────────▶ X3
```

with

- `X1 ~ N(0, 1)`;
- `X2 = X1^2 + epsilon2`, where `epsilon2 ~ t(df=1)`;
- `X3 = cos(X2 - 1) + exp(X1) - 2 + tanh(X2 - X1) + epsilon3`, where `epsilon3 ~ N(0, 1)`.

The direct `X1 -> X3` edge means that `X1` and `X3` need not be conditionally independent given `X2`.

## Run the example

From the repository root:

```bash
python examples/discovery_data.py
python examples/discovery.py
```

The default data path and output directory are resolved relative to `examples/`, so the commands also work when launched from another working directory.

For a quicker smoke run, reduce the training and Monte Carlo settings:

```bash
python examples/discovery.py --n_epochs 1000 --mc_samples 100 --D 100
```

Use `--device cpu` when CUDA is unavailable.

## What is tested

The manuscript narrows the candidate graph set with three checks:

1. Marginal dependence between `X1` and `X3` excludes the graph with no path between them.
2. An additive-noise residual check based on `X2 - E[X2 | X1, X3]` tests the candidate in which `X2` has parents `X1` and `X3`.
3. A conditional-independence test of `X1 ⟂ X3 | X2` evaluates the chain-only graph. Rejection excludes that graph and is consistent with the direct `X1 -> X3` edge in G2.

The script calls the residual candidate “Model 1” and the chain null “Model 2.” These are candidate models, not the true data-generating graph.

## Outputs

```text
examples/
├── data/discovery.csv
├── ckpt/discovery_model1/
├── ckpt/discovery_model2/
├── results/discovery_model1/conditional_expectations.csv
└── results/discovery_model2/
    ├── tau_samples.csv
    └── tau_observed.csv
```

For Model 2, a small p-value rejects `X1 ⟂ X3 | X2`. Checkpoints, generated data, and results are ignored by Git.

See `discovery.ipynb` for visualization after the computation finishes. Run `python examples/discovery.py --help` for the full parameter list.
