# src/hparams.py
"""
Hyperparameter search methods for domain generalization experiments.
"""

import numpy as np
from domainbed.hparams_registry import random_hparams as sample_hparams_domainbed


class RandomSearch:
    name = 'random'

    @staticmethod
    def get_hparams(algorithm_name, dataset_name, n_hparams, backbone='resnet50'):
        configs = []
        for seed in range(n_hparams):
            hp = sample_hparams_domainbed(algorithm_name, dataset_name, seed)
            if backbone == 'clip':
                hp['use_clip'] = True
            configs.append({
                'hparams_seed': seed,
                'hparams':      dict(hp),
            })
        return configs


class GridSearch:
    name = 'grid'

    @staticmethod
    def get_hparams(algorithm_name, dataset_name, n_hparams, backbone='resnet50'):
        raise NotImplementedError("Grid search not yet implemented.")


class BayesianSearch:
    name = 'bayesian'

    @staticmethod
    def get_hparams(algorithm_name, dataset_name, n_hparams, backbone='resnet50'):
        raise NotImplementedError("Bayesian search not yet implemented.")


HP_SEARCH_METHODS = {
    'random':   RandomSearch,
    'grid':     GridSearch,
    'bayesian': BayesianSearch,
}