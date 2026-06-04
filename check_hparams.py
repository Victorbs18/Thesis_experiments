import json
import numpy as np
import sys
sys.path.insert(0, 'DomainBed')
from domainbed.lib.query import Q
from domainbed.model_selection import IIDAccuracySelectionMethod, OracleSelectionMethod

with open('./results/coloredmnist/test_env2/cnn/random/records.json') as f:
    records = json.load(f)

q = Q(records)
for algo in ['ERM', 'IRM']:
    algo_records = q.filter(lambda r: r['algorithm'] == algo)
    for method in [IIDAccuracySelectionMethod, OracleSelectionMethod]:
        accs = method.hparams_accs(algo_records)
        if len(accs):
            best = accs[0][1].sorted(lambda r: r['step'])[-1]
            print(f"{algo} [{method.name}]: "
                  f"env2_out={best['env2_out_acc']:.3f} "
                  f"hp_seed={best['args']['hparams_seed']}")
