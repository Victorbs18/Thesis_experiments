# src/networks.py
"""
Network architectures for domain generalization experiments.

For standard backbones (CNN, ResNet50) DomainBed's Featurizer is used
directly — it auto-selects based on input shape:
    (2, 28, 28)  → MNIST_CNN
    (3, 224, 224) → ResNet50 (default)

For CLIP, a custom CLIPFeaturizer is defined here and DomainBed's
Featurizer is monkey-patched via patch_domainbed_for_clip().
"""

import torch
import torch.nn as nn
from domainbed.networks import MNIST_CNN, ResNet, Featurizer, Classifier


class CLIPFeaturizer(nn.Module):
    """
    CLIP ViT-B/32 image encoder.
    Loads OpenAI pretrained weights.
    Output: 512-dim feature vector per image.

    Notes:
        - CLIP uses float16 by default — converted to float32 here
        - Uses only the visual encoder (image side of CLIP)
        - Dropout applied after encoding (matches ResNet setup)
    """
    def __init__(self, hparams):
        super().__init__()
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        model, _       = clip.load("ViT-B/32", device='cpu')
        self.model     = model.visual.float()  # float32
        self.n_outputs = 512
        self.dropout   = nn.Dropout(hparams.get('resnet_dropout', 0.0))

    def forward(self, x):
        return self.dropout(self.model(x))


def patch_domainbed_for_clip():
    """
    Monkey-patch DomainBed's Featurizer to return CLIPFeaturizer
    for 224x224 inputs when hparams['use_clip'] is True.

    Call this before instantiating any DomainBed algorithm with CLIP.
    Stores original Featurizer so it can be restored with unpatch.
    """
    import domainbed.networks as db_networks

    if hasattr(db_networks, '_original_featurizer'):
        return  # already patched

    db_networks._original_featurizer = db_networks.Featurizer

    def patched_featurizer(input_shape, hparams):
        if (input_shape[1:3] == (224, 224) and
                hparams.get('use_clip', False)):
            return CLIPFeaturizer(hparams)
        return db_networks._original_featurizer(input_shape, hparams)

    db_networks.Featurizer = patched_featurizer
    print("  DomainBed Featurizer patched for CLIP")


def unpatch_domainbed():
    """Restore original DomainBed Featurizer."""
    import domainbed.networks as db_networks
    if hasattr(db_networks, '_original_featurizer'):
        db_networks.Featurizer = db_networks._original_featurizer
        del db_networks._original_featurizer