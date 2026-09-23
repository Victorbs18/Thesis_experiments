# src/hparams.py
"""
Hyperparameter search methods for domain generalization experiments.
"""

import numpy as np
from domainbed.hparams_registry import random_hparams as sample_hparams_domainbed


# src/hparams.py

def _add_csd_hparams(hp, seed):
    rng = np.random.RandomState(seed)
    hp.setdefault('csd_lambda1',        1.0)   # force equal lambdas
    hp.setdefault('csd_lambda2',        1.0)   # pure concept shift mode
    hp.setdefault('d_steps_per_g_step', 1)     # simpler alternation
    hp.setdefault('lr_d',               1e-4)
    hp.setdefault('weight_decay_d',     0.0)
    hp.setdefault('beta1',              0.5)
    return hp


# Maps algorithm name to its custom hparam injector, if any
CUSTOM_HPARAMS = {
    'CSD': _add_csd_hparams,
}


class RandomSearch:
    name = 'random'

    @staticmethod
    def get_hparams(algorithm_name, dataset_name, n_hparams, backbone='resnet50'):
        configs = []
        for seed in range(n_hparams):
            hp = sample_hparams_domainbed(algorithm_name, dataset_name, seed)

            # Inject algorithm-specific hparams not in domainbed registry
            if algorithm_name in CUSTOM_HPARAMS:
                hp = CUSTOM_HPARAMS[algorithm_name](hp, seed)

            if backbone == 'clip':
                hp['use_clip'] = True
                rng = np.random.RandomState(seed)
                hp['lr'] = float(10 ** rng.uniform(-6, -4.5))

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