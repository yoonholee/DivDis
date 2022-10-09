import os
import sys

import numpy as np
import torch
import torch.nn as nn

from models.initializer import initialize_model
from utils import move_to
from algorithms.single_model_algorithm import SingleModelAlgorithm

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from divdis import DivDisLoss, to_probs


class MultiHeadModel(nn.Module):
    def __init__(self, featurizer, classifier, heads=2):
        super().__init__()
        self.heads = heads
        self.featurizer = featurizer
        in_dim, out_dim = classifier.in_features, classifier.out_features * self.heads
        self.heads_classifier = nn.Linear(in_dim, out_dim)

        self.needs_y = featurizer.needs_y

    def forward(self, x):
        features = self.featurizer(x)
        outputs = self.heads_classifier(features)
        return outputs


class DivDis(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        multihead_model = MultiHeadModel(featurizer, classifier, config.divdis_heads)
        self.training_mode = True
        self.best_head_idx = 0
        self.diversity_weight = config.divdis_diversity_weight

        # initialize module
        super().__init__(
            config=config,
            model=multihead_model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.loss_fn = DivDisLoss(heads=config.divdis_heads)
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
        # package the results
        results = {"g": g, "y_true": y_true, "metadata": metadata}

        pred = self.get_model_output(x, None)
        preds_chunked = torch.chunk(pred, self.model.heads, dim=-1)
        for i in range(self.model.heads):
            results[f"y_pred_{i}"] = preds_chunked[i]
        results["y_pred"] = preds_chunked[self.best_head_idx]

        if unlabeled_batch is not None:
            x_unlab, metadata_unlab = unlabeled_batch
            x_unlab = move_to(x_unlab, self.device)
            g_unlab = move_to(
                self.grouper.metadata_to_group(metadata_unlab), self.device
            )
            pred_unlab = self.get_model_output(x_unlab, None)

            results["unlabeled_metadata"] = metadata_unlab
            results["unlabeled_g"] = g_unlab
            results["unlabeled_y_pred"] = pred_unlab

        # During evaluation, accumulate corrects and disagreements for head selection.
        if self.training == False:
            if self.training_mode == True:
                if hasattr(self, "all_corrects"):  # re-compute best head
                    all_corrects = np.concatenate(self.all_corrects)
                    all_disagreements = np.concatenate(self.all_disagreements)

                    query_idxs = all_disagreements.argsort()[-1000:]
                    query_corrects = all_corrects[query_idxs]
                    accs = np.float32(query_corrects).mean(axis=0)
                    self.best_head_idx = int(accs.argsort()[-1])
                    print("Recomputed best head:", self.best_head_idx)
                self.all_corrects = []
                self.all_disagreements = []

            corrects = [p.argmax(-1) == y_true for p in preds_chunked]
            corrects = torch.stack(corrects, -1)
            corrects = corrects.detach().cpu().numpy()
            self.all_corrects.append(corrects)

            probs = to_probs(pred, self.model.heads)
            diff = probs.unsqueeze(1) - probs.unsqueeze(2)
            disagreement = diff.abs().mean([-3, -2, -1])
            disagreement = disagreement.detach().cpu().numpy()
            self.all_disagreements.append(disagreement)
        self.training_mode = self.training

        return results

    def objective(self, results):
        # Labeled loss
        classification_losses = [
            self.loss.compute(
                results[f"y_pred_{i}"], results["y_true"], return_dict=False
            )
            for i in range(self.model.heads)
        ]
        classification_loss = sum(classification_losses)

        if "unlabeled_y_pred" in results.keys():
            repulsion_loss = self.loss_fn(results["unlabeled_y_pred"])
        else:
            repulsion_loss = 0.0

        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(results, "repulsion_loss", repulsion_loss)

        return classification_loss + repulsion_loss * self.diversity_weight
