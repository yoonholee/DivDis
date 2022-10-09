import torch
import torch.nn as nn
from models.initializer import initialize_model
from utils import move_to

from algorithms.single_model_algorithm import SingleModelAlgorithm


def get_mlp(in_dim, dims):
    module_list = []
    past_dim = in_dim
    for d in dims:
        module_list.extend([nn.Linear(past_dim, d), nn.ReLU()])
        past_dim = d
    return nn.Sequential(*module_list[:-1])


class LatentModel(nn.Module):
    def __init__(self, featurizer, classifier, config):
        super().__init__()
        self.needs_y = False
        self.featurizer = featurizer
        self.zdim = config.l_divdis_zdim
        self.cls_width = config.l_divdis_cls_width
        self.cls_depth = config.l_divdis_cls_depth
        self.inf_width = config.l_divdis_inf_width
        self.inf_depth = config.l_divdis_inf_depth

        self.hdim, self.ydim = classifier.in_features, classifier.out_features
        self.classifier = get_mlp(
            self.zdim + self.hdim, [self.cls_width] * self.cls_depth + [self.ydim]
        )
        self.inferrer = get_mlp(
            self.hdim, [self.inf_width] * self.inf_depth + [self.zdim * self.ydim]
        )

    def sample_latent(self, shape, device):
        """Sample z_dim-dimensional latents, uniform [-1, 1]."""
        return torch.rand((*shape, self.zdim), device=device) * 2 - 1

    def forward_dict_with_z(self, x, z):
        h = self.featurizer(x)
        z_and_h = torch.cat([z, h], dim=-1)
        logits = self.classifier(z_and_h)
        return {"z": z, "h": h, "logits": logits}

    def forward_dict(self, x):
        z = self.sample_latent((x.shape[0],), x.device)
        return self.forward_dict_with_z(x, z)

    def forward(self, x):
        outputs = self.forward_dict(x)
        return outputs["logits"]

    def infer(self, h):
        z_preds = self.inferrer(h)
        return z_preds.reshape(-1, self.zdim, self.ydim)

    @staticmethod
    def repulsion_loss(z_preds, z, y_logits):
        y_probs = y_logits.softmax(-1)
        z_diff = z_preds - z.unsqueeze(-1)  # [B, Z, C]
        z_mse = z_diff.pow(2).sum(1)  # [B, Z, C]
        loss_repulsion = (z_mse * y_probs).sum(-1).mean()
        return loss_repulsion


class LatentDivDis(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        latent_model = LatentModel(featurizer, classifier, config=config)
        self.training_mode = True
        self.best_head_idx = 0
        self.reconst_weight = config.l_divdis_reconst_weight
        # initialize module
        super().__init__(
            config=config,
            model=latent_model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        # additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("repulsion_loss")

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (x, y, m): a batch of data yielded by data loaders
            - unlabeled_batch: examples (x, m)
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_y_pred (Tensor): model output on unlabeled batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - metadata (Tensor): metadata for batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
        """
        # Labeled examples
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        results = {"g": g, "y_true": y_true, "metadata": metadata}

        y_pred = self.model.forward(x)
        results["y_pred"] = y_pred

        if unlabeled_batch is not None:
            x_unlab, metadata_unlab = unlabeled_batch
            x_unlab = move_to(x_unlab, self.device)
            g_unlab = move_to(
                self.grouper.metadata_to_group(metadata_unlab), self.device
            )
            results_unlabeled = {
                "unlabeled_metadata": metadata_unlab,
                "unlabeled_g": g_unlab,
            }
            results.update(results_unlabeled)

            out_unlab = self.model.forward_dict(x_unlab)
            z_preds = self.model.infer(out_unlab["h"])
            results_latent_divdis = {
                "z_unlab": out_unlab["z"],
                "h_unlab": out_unlab["h"],
                "y_pred_unlab": out_unlab["logits"],
                "z_pred_unlab": z_preds,
            }
            results.update(results_latent_divdis)

        return results

    def objective(self, results):
        loss_cls = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        used_unlabeled = "y_pred_unlab" in results.keys()
        if used_unlabeled:
            loss_repulsion = self.model.repulsion_loss(
                results["z_pred_unlab"], results["z_unlab"], results["y_pred_unlab"]
            )
        else:
            loss_repulsion = 0.0

        loss = loss_cls + self.reconst_weight * loss_repulsion

        self.save_metric_for_logging(results, "classification_loss", loss_cls)
        self.save_metric_for_logging(results, "repulsion_loss", loss_repulsion)
        return loss
