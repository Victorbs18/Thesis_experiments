"""
Export two thesis figures:
  1. results/rotated_mnist_digit6.png  — digit 6 at each rotation (0°..75°)
  2. results/colored_mnist_digit6.png  — digit 6 in each ColoredMNIST env
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DomainBed'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from domainbed.datasets import RotatedMNIST as DB_RotatedMNIST

DATA_DIR     = './data'
TARGET_DIGIT = 6
os.makedirs('./results', exist_ok=True)

# ──────────────────────────────────────────────
# Figure 1 — RotatedMNIST
# ──────────────────────────────────────────────
ROTATIONS = ['0°', '15°', '30°', '45°', '60°', '75°']
db = DB_RotatedMNIST(DATA_DIR, test_envs=[5], hparams={})

rot_imgs = []
for env in db.datasets:
    for img, label in env:
        if int(label) == TARGET_DIGIT:
            arr = img.numpy()
            if arr.ndim == 3:
                arr = arr[0]
            rot_imgs.append(arr)
            break

n = len(rot_imgs)
fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2.0))
for ax, img, name in zip(axes, rot_imgs, ROTATIONS):
    ax.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(name, fontsize=10)
    ax.axis('off')
fig.suptitle(f'RotatedMNIST', fontsize=11)
plt.tight_layout()
out1 = './results/rotated_mnist_digit6.png'
plt.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved {out1}")
plt.close()

# ──────────────────────────────────────────────
# Figure 2 — ColoredMNIST
# ──────────────────────────────────────────────
mnist_train = MNIST(DATA_DIR, train=True,  download=True)
mnist_test  = MNIST(DATA_DIR, train=False, download=True)

images_raw = torch.cat([mnist_train.data, mnist_test.data]).float()
labels_raw = torch.cat([mnist_train.targets, mnist_test.targets])

rng  = torch.Generator(); rng.manual_seed(0)
perm = torch.randperm(len(images_raw), generator=rng)
images_raw = images_raw[perm]
labels_raw = labels_raw[perm]

def _bernoulli(p, size):
    return (torch.rand(size) < p).float()

def _xor(a, b):
    return (a - b).abs()

def _color_dataset(images, labels, environment):
    labels = (labels < 5).float()
    labels = _xor(labels, _bernoulli(0.25, len(labels)))
    colors = _xor(labels, _bernoulli(environment, len(labels)))
    images = torch.stack([images, images], dim=1)
    images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0
    x = images.float().div_(255.0)
    return x  # (N, 2, H, W)

environments = [0.1, 0.2, 0.9]
env_names    = ['+90%', '+80%', '−90%']

colored_imgs = []
for env_offset, env_noise in enumerate(environments):
    imgs_env   = images_raw[env_offset::3]
    labels_env = labels_raw[env_offset::3]

    # Find position of first digit-6 before colorization
    idx6 = (labels_env == TARGET_DIGIT).nonzero(as_tuple=True)[0][0]

    # Colorize exactly as in the real dataset (same torch state, no manual seed)
    two_ch_env = _color_dataset(imgs_env, labels_env.clone(), env_noise)
    two_ch = two_ch_env[idx6]  # (2, H, W)

    rgb = np.zeros((*two_ch.shape[1:], 3), dtype=np.float32)
    rgb[:, :, 0] = two_ch[0].numpy()
    rgb[:, :, 1] = two_ch[1].numpy()
    colored_imgs.append(rgb)

fig, axes = plt.subplots(1, 3, figsize=(4.5, 2.0))
for ax, img, name in zip(axes, colored_imgs, env_names):
    ax.imshow(img, interpolation='nearest')
    ax.set_title(name, fontsize=10)
    ax.axis('off')
fig.suptitle(f'ColoredMNIST', fontsize=11)
plt.tight_layout()
out2 = './results/colored_mnist_digit6.png'
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved {out2}")
plt.close()
