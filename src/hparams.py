# src/hparams.py

"""
Hyperparameter search methods for domain generalization experiments.

All search methods return a list of dicts:
    [
        {'hparams_seed': int, 'hparams': dict},
        ...
    ]
This list is consumed by run_sweep() in train.py.
"""

import numpy as np
from domainbed.hparams_registry import random_hparams as sample_hparams_domainbed


class RandomSearch:
    """
    Random search over HP space (DomainBed)
    """
    name = 'random'

    @staticmethod
    def get_hparams(algorithm_name, dataset_name, n_hparams):
        configs = []
        for seed in range(n_hparams):
            hp  = sample_hparams_domainbed(algorithm_name, dataset_name,seed)
            configs.append({
                'hparams_seed': seed,
                'hparams':      dict(hp),
            })
        return configs


class GridSearch:
    """
    Grid search over HP space.
    """
    name = 'grid'

    @staticmethod
    def get_hparams(algorithm_name, dataset_name, n_hparams):
        raise NotImplementedError(
            "Grid search not yet implemented. "
            "Define a grid for your algorithm/dataset combination."
        )


class BayesianSearch:
    """
    """
    name = 'bayesian'

    @staticmethod
    def get_hparams(algorithm_name, dataset_name, n_hparams):
        raise NotImplementedError(
            "Bayesian search is sequential and cannot precompute configs. "
            "Use run_sweep_bayesian() instead of run_sweep()."
        )


HP_SEARCH_METHODS = {
    'random':   RandomSearch,
    'grid':     GridSearch,
    'bayesian': BayesianSearch,
}
