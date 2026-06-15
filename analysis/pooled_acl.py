"""
Pooled accuracy-on-the-line for PACS, combining ResNet50 + CLIP backbones.

Mimics Salaudeen et al.'s setup more closely: pool classifiers across
multiple architectures (here: ResNet50 and CLIP ViT-B/32) and compute
a single R value per algorithm.

Usage:
    python pooled_acl.py
"""
import json
import numpy as np
from scipy.special import ndtri as probit
from scipy.stats import pearsonr, linregress


def compute_id_acc(record, test_env_idx, n_envs):
    train_accs = [
        record[f'env{i}_out_acc']
        for i in range(n_envs)
        if i != test_env_idx
    ]
    return np.mean(train_accs)


def compute_ood_acc(record, test_env_idx):
    return record[f'env{test_env_idx}_out_acc']


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
    id_accs  = np.array([p['id_acc'] for p in points])
    ood_accs = np.array([p['ood_acc'] for p in points])

    eps = 1e-6
    id_p  = probit(np.clip(id_accs, eps, 1 - eps))
    ood_p = probit(np.clip(ood_accs, eps, 1 - eps))

    R, pval = pearsonr(id_p, ood_p)
    reg = linregress(id_p, ood_p)

    flag = '✓ well-specified' if R < 0.3 else '✗ misspecified'
    print(f"{label:<30} n={len(points):3d}  R={R:+.3f}  "
          f"slope={reg.slope:.3f}  intercept={reg.intercept:.3f}  "
          f"p={pval:.2e}  se={reg.stderr:.3f}  {flag}")
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