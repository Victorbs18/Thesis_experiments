##  Thesis Experiments: Agreement-based Diagnostics for Domain Generalization

### Setup

```bash
git clone --recurse-submodules <this repo>
cd Thesis_experiments
conda env create -f environment.yaml

# Local fixes to the DomainBed submodule (seeded RotatedMNIST shuffle,
# CSD / tabular hyperparameters). Base commit: b93c22a1cfc3b2428398272c1a116c8de1f4139e
git -C DomainBed apply ../patches/domainbed.patch
```

The patch must be applied on every fresh clone: changes inside the submodule
are not part of this repository's history.
