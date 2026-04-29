# Thesis Experiments — Agreement-based Diagnostics for Domain Generalization

**Author:** Victor  
**Supervisor:** Werner Zellinger — LIT AI Lab, JKU Linz  
**Thesis:** Agreement-based diagnostics for domain generalization: predicting when invariant learning outperforms ERM

---

## Structure

```
thesis_experiments/
└── colored_mnist/
    ├── data.py          # Colored MNIST data loading and environment construction
    ├── models.py        # MLP model and training (ERM, IRM, VREx)
    ├── agreement.py     # Agreement, entropy, AGL line fitting
    ├── agl_2x2.py       # Main experiment: 2x2 proximity/diversity matrix
    └── results/         # Output plots and JSON (auto-created)
```

---

## Experiments

### Experiment: 2x2 Proximity × Diversity Matrix

Replicates the 2x2 design from the practical work (Experiment 5) and adds
the IRM-ERM agreement diagnostic on top. Tests whether the agreement signal
correctly identifies when IRM helps vs when it collapses, across four
qualitatively different environment configurations.

```
Config  e_train     Diversity  Proximity  Expected decision
A       {0.1, 0.2}  Low        Low        AGREE (IRM collapses non-oracle)
B       {0.7, 0.8}  Low        High       AGREE (ERM already solves task)
C       {0.1, 0.5}  High       Low        DIVERGE (IRM genuinely helps)
D       {0.1, 0.8}  High       High       WEAK DIVERGE (marginal IRM advantage)
```

**Run:**
```bash
python agl_2x2.py --n_trials 20 --n_seeds 3 --device cpu
```

---

## Setup

Same conda environment as practical work:
```bash
conda activate irm_reproduction
pip install scipy matplotlib  # if not already installed
```

---

## Relationship to practical work

The `colored_mnist/` folder here is intentionally separate from
`Domain_Generalization_Analysis/colored_mnist/`. The practical work
reproduces the IRM paper. This repository develops the thesis contribution
on top of those findings. Code is reused and simplified where possible
but kept independent to avoid mixing concerns.
