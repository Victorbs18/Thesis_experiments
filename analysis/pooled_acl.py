"""
Pooled accuracy-on-the-line for PACS, combining ResNet50 + CLIP backbones.

Mimics Salaudeen et al.'s setup more closely: pool classifiers across
multiple architectures (here: ResNet50 and CLIP ViT-B/32) and compute
a single R value per algorithm.

Usage:
    python pooled_acl.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import compute_id_acc, compute_ood_acc, fit_line


def load_points(records_path, test_env_idx, n_envs, algo, backbone_label):
    with open(records_path) as f:
        records = json.load(f)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        if r['algorithm'] != algo:
            continue
        hp_seed = r['args']['hparams_seed']
        id_acc  = compute_id_acc(r, test_env_idx, n_envs)
        ood_acc = compute_ood_acc(r, test_env_idx)
        grouped[hp_seed].append((id_acc, ood_acc))

    points = []
    for hp_seed, trial_pairs in sorted(grouped.items()):
        id_mean  = np.mean([p[0] for p in trial_pairs])
        ood_mean = np.mean([p[1] for p in trial_pairs])
        points.append({
            'hp_seed': hp_seed,
            'id_acc': id_mean,
            'ood_acc': ood_mean,
            'backbone': backbone_label,
        })
    return points


def compute_acl(points, label):
    id_accs  = [p['id_acc'] for p in points]
    ood_accs = [p['ood_acc'] for p in points]

    line = fit_line(id_accs, ood_accs)
    R, pval = line['R'], line['p_value']

    flag = '✓ well-specified' if R < 0.3 else '✗ misspecified'
    print(f"{label:<30} n={len(points):3d}  R={R:+.3f}  "
          f"slope={line['slope']:.3f}  intercept={line['intercept']:.3f}  "
          f"p={pval:.2e}  se={line['std_error']:.3f}  {flag}")
    return R, pval


for algo in ['ERM', 'IRM']:
    print(f"\n=== {algo} ===")

    resnet_points = load_points(
        'results/pacs/test_env0/resnet50/random/records.json',
        test_env_idx=0, n_envs=4, algo=algo, backbone_label='resnet50')

    clip_points = load_points(
        'results/pacs/test_env0/clip/random/records.json',
        test_env_idx=0, n_envs=4, algo=algo, backbone_label='clip')

    compute_acl(resnet_points, 'ResNet50 only')
    compute_acl(clip_points, 'CLIP only')
    compute_acl(resnet_points + clip_points, 'Pooled (ResNet50 + CLIP)')