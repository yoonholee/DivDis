import torch
from models.initializer import initialize_model
from utils import move_to

from algorithms.single_model_algorithm import SingleModelAlgorithm


class BN(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        model = initialize_model(config, d_out=d_out)
        model = model.to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.logged_fields.append("classification_loss")

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pseudo (Tensor): pseudolabels on the unlabeled batch, already thresholded
                - unlabeled_y_pred (Tensor): model output on the unlabeled batch, already thresholded
        """
        # Labeled examples
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        results = {"g": g, "y_true": y_true, "metadata": metadata}
        if unlabeled_batch is not None:
            x_unlab, _ = unlabeled_batch
            x_unlab = move_to(x_unlab, self.device)
            # run unlabeled batch forward and discard
            with torch.no_grad():
                self.get_model_output(x_unlab, None) 

        results["y_pred"] = self.get_model_output(x, y_true)
        return results

    def objective(self, results):
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        return classification_loss
