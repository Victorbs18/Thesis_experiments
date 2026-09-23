# src/algorithms.py

from domainbed.algorithms import (
    ERM, IRM, GroupDRO, CORAL, DANN, VREx, Algorithm)
from domainbed import networks
import torch
import torch.nn.functional as F


class CSD(Algorithm):
    """
    Concept Shift Discriminator (CSD)

    Targets concept shift by directly penalizing the cross-environment
    variance of the feature-label covariance. For each feature dimension k,
    computes Cov^e(Z_k, Y) in each environment and penalizes its variance
    across environments — the direct signature of concept shift.

    This replaced an earlier two-discriminator gap approach (D^Z vs D^ZY),
    which was too noisy in practice because the gap between two
    simultaneously-trained networks is dominated by optimization noise
    rather than concept shift signal.

    Full objective:
        L = L_CE(h(Z), Y)                    # task loss
          + lambda_dann * L_dann              # optional DANN term
          + lambda_cs * CS_penalty(Z, Y, e)  # concept shift penalty

    CS_penalty = mean over feature dims of Var_e[Cov^e(Z_k, Y)]^2
    Quadratic amplification: penalizes large cross-environment variance
    much more heavily than small variance.

    Note: Cov^e(Z_k, Y) treats Y as a scalar, so this is only a clean
    signal for binary classification (e.g. ColoredMNIST) where Y in {0,1}
    is itself a meaningful indicator. For multi-class datasets the raw
    integer label has no ordinal meaning, so the covariance would need to
    be computed per-class (e.g. one-hot Y) before this generalizes.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CSD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.num_domains = num_domains
        self.num_classes = num_classes

        # Featurizer and classifier
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )

        # Optional DANN discriminator on Z alone.
        # Controlled by lambda_dann: set to 0 for pure concept-shift mode.
        self.discriminator = networks.MLP(
            self.featurizer.n_outputs,
            num_domains,
            self.hparams
        )

        # Featurizer + classifier optimizer
        self.gen_opt = torch.optim.Adam(
            list(self.featurizer.parameters()) +
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            betas=(self.hparams["beta1"], 0.9)
        )

        # Discriminator optimizer
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9)
        )

    def _concept_shift_penalty(self, features_per_env, labels_per_env):
        """
        Computes cross-environment variance of feature-label covariance.

        For each feature dimension k:
            cov^e_k = Cov(Z_k, Y) in environment e
        Penalty = mean_k[ Var_e[cov^e_k]^2 ]

        Quadratic amplification: large cross-environment variance in
        feature-label relationship is penalized much more than small variance.

        features_per_env: list of (N_e, d) tensors, one per environment
        labels_per_env:   list of (N_e,) tensors, one per environment
        returns: scalar penalty
        """
        covs = []
        for z_e, y_e in zip(features_per_env, labels_per_env):
            # Center features and labels within environment
            z_centered = z_e - z_e.mean(dim=0, keepdim=True)      # (N_e, d)
            y_float = y_e.float()
            y_centered = y_float - y_float.mean()                  # (N_e,)

            # Per-dimension covariance: (d,)
            cov_e = (z_centered * y_centered.unsqueeze(1)).mean(dim=0)
            covs.append(cov_e)

        # Stack: (num_envs, d)
        covs = torch.stack(covs, dim=0)

        # Cross-environment variance per feature dimension: (d,)
        cov_var = covs.var(dim=0)

        # Quadratic amplification and mean over feature dimensions
        penalty = (cov_var ** 2).mean()

        return penalty

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        all_e = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        d_steps_per_g = self.hparams["d_steps_per_g_step"]

        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:
            # --- Discriminator update ---
            all_z = self.featurizer(all_x).detach()
            disc_loss = F.cross_entropy(self.discriminator(all_z), all_e)

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()

            return {"disc_loss": disc_loss.item()}

        else:
            # --- Generator update ---
            all_z = self.featurizer(all_x)

            # Task loss
            task_loss = F.cross_entropy(self.classifier(all_z), all_y)

            # Optional DANN term
            disc_loss = F.cross_entropy(self.discriminator(all_z), all_e)

            # Concept shift penalty: variance of feature-label covariance
            # across environments, amplified quadratically
            features_per_env = []
            labels_per_env   = []
            idx = 0
            for x, y in minibatches:
                n = x.shape[0]
                features_per_env.append(all_z[idx:idx+n])
                labels_per_env.append(y)
                idx += n

            cs_penalty = self._concept_shift_penalty(
                features_per_env, labels_per_env)

            lambda_dann = self.hparams["csd_lambda1"]
            lambda_cs   = self.hparams["csd_lambda2"]

            gen_loss = (
                task_loss
                + lambda_dann * disc_loss
                + lambda_cs   * cs_penalty
            )

            # --- DEBUG ---
            if self.update_count.item() % 100 == 0:
                print(f"  [step {self.update_count.item()}] "
                      f"cs_penalty={cs_penalty.item():.8f} "
                      f"disc_loss={disc_loss.item():.4f} "
                      f"task_loss={task_loss.item():.4f}")
            # --- END DEBUG ---

            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()

            return {
                "task_loss":  task_loss.item(),
                "disc_loss":  disc_loss.item(),
                "cs_penalty": cs_penalty.item(),
                "gen_loss":   gen_loss.item()
            }

    def predict(self, x):
        return self.classifier(self.featurizer(x))


ALGORITHMS = {
    'ERM':      ERM,
    'IRM':      IRM,
    'GroupDRO': GroupDRO,
    'CORAL':    CORAL,
    'DANN':     DANN,
    'VREx':     VREx,
    'CSD':      CSD,
}