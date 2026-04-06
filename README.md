# Zipfian grokking

This repository contains code and essays for an ongoing research program on eliciting generalization in neural networks, centered on a toy problem we call *Zipfian grokking*: modular arithmetic with Zipf-weighted loss. The Zipfian weighting creates a controllable tension between memorization and generalization, making it a useful testbed for new methods of pushing models toward deeper abstractions.

## Essays

1. **[Zipfian grokking](essays/zipfian_grokking.md)** — Introduces the toy problem. Weighting the loss by Zipf's Law turns the Fourier solution into a saddle point, producing Sisyphean dynamics: models repeatedly discover the correct solution and are expelled from it. We characterize the instability as a relaxation oscillator and confirm it experimentally.

2. **[Interaction as supervision](essays/interaction_as_supervision.md)** — Uses the same toy problem to test whether auxiliary inverse dynamics objectives can stabilize generalization under Zipfian pressure. Inverse models create a strict selection pressure for Fourier structure that forward models do not. We also show that a pairwise tournament can autonomously discover which transformation families are most learnable, without being told anything about Fourier structure.

## Repository structure

```
├── essays/
│   ├── zipfian_grokking.md              # Essay 1: the toy problem
│   └── interaction_as_supervision.md     # Essay 2: ICM and symmetry discovery
├── train_grokking_zipfian.py             # Core Zipfian grokking training
├── run_ablation.py                       # Multi-seed ablation study
├── train_uniform_then_zipf.py            # Two-phase saddle point test
├── interaction_as_supervision/
│   ├── train_inverse_primary.py          # ICM inverse-primary training
│   ├── train_learned_policy.py           # Policy-guided transform selection
│   ├── train_round_robin_cocktail.py     # Fixed-mixture cocktail training
│   ├── analyze_tournament.py             # Tournament analysis and ranking
│   └── run_tournament.sh                 # Orchestrates the full tournament
└── figures/                              # Pre-computed visualizations
```

## Requirements

```
jax
jaxlib
flax
optax
numpy
matplotlib
tqdm
```

## Reproducing the experiments

### The toy problem and its dynamics

**Zipfian ablation** (Figure 1 in essay 1): train at multiple Zipf exponents and compare dynamics.

```bash
python run_ablation.py \
  --zipf_exponents 0.0 1.0 1.5 2.0 \
  --weighted_loss \
  --n_epochs 1000000 \
  --output_dir ./ablation_results
```

**Single Zipfian run:**

```bash
python train_grokking_zipfian.py \
  --zipf_exponent 1.5 \
  --weighted_loss \
  --n_epochs 500000 \
  --save_dir ./output
```

**Saddle point test** (uniform pretrain, then Zipfian finetune):

```bash
python train_uniform_then_zipf.py \
  --zipf_exponent 1.5 \
  --n_epochs_phase1 50000 \
  --n_epochs_phase2 200000 \
  --save_dir ./saddle_test
```

### Inverse dynamics and symmetry discovery

**Inverse-primary training** (the core ICM experiment):

```bash
# Inverse-only (recommended):
python interaction_as_supervision/train_inverse_primary.py \
  --zipf_exponent 1.5 \
  --beta_inverse 0.1 \
  --beta_forward 0.0 \
  --n_epochs 2000000 \
  --save_dir ./results_inverse_only

# Forward-only (for comparison):
python interaction_as_supervision/train_inverse_primary.py \
  --zipf_exponent 1.5 \
  --beta_inverse 0.0 \
  --beta_forward 0.1 \
  --n_epochs 2000000 \
  --save_dir ./results_forward_only

# Full ICM (both):
python interaction_as_supervision/train_inverse_primary.py \
  --zipf_exponent 1.5 \
  --beta_inverse 0.1 \
  --beta_forward 0.1 \
  --n_epochs 2000000 \
  --save_dir ./results_full_icm

# Baseline (task only):
python interaction_as_supervision/train_inverse_primary.py \
  --zipf_exponent 1.5 \
  --beta_inverse 0.0 \
  --beta_forward 0.0 \
  --n_epochs 2000000 \
  --save_dir ./results_baseline
```

**Transform tournament** (autonomous symmetry discovery):

```bash
# Run the full tournament (round-robin + championship):
bash interaction_as_supervision/run_tournament.sh ./results_tournament

# Or run a single pairwise match (e.g., translation vs random):
python interaction_as_supervision/train_learned_policy.py \
  --enabled_transforms=0,3 \
  --policy_guided_calibration \
  --n_epochs 500000 \
  --save_dir ./results_translation_vs_random
```

**Round-robin cocktail** (fixed mixture from tournament results):

```bash
python interaction_as_supervision/train_round_robin_cocktail.py \
  --round_robin_summary ./results_tournament/round_robin/summary.json \
  --n_epochs 2000000 \
  --save_dir ./results_cocktail
```

## Citation

```
@article{gilley2026zipfiangrokking,
  title   = {Zipfian grokking},
  author  = {Gilley, Jasper},
  year    = {2026},
  month   = {March},
  url     = {https://jagilley.github.io/zipfian-grokking.html}
}

@article{gilley2026interactionsupervision,
  title   = {Interaction as supervision},
  author  = {Gilley, Jasper},
  year    = {2026},
  month   = {March},
  url     = {https://jagilley.github.io/interaction-as-supervision.html}
}
```
