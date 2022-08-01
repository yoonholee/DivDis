import torch
from einops import rearrange
from torch import nn


def to_probs(logits, heads):
    """
    Converts logits to probabilities.
    Input must have shape [batch_size, heads * classes].
    Output will have shape [batch_size, heads, classes].
    """

    B, N = logits.shape
    if N == heads:  # Binary classification; each head outputs a single scalar.
        preds = logits.sigmoid().unsqueeze(-1)
        probs = torch.cat([preds, 1 - preds], dim=-1)
    else:
        logits_chunked = torch.chunk(logits, heads, dim=-1)
        probs = torch.stack(logits_chunked, dim=1).softmax(-1)
    B, H, D = probs.shape
    assert H == heads
    return probs


def get_disagreement_scores(logits, heads, mode="l1"):
    probs = to_probs(logits, heads)
    if mode == "l1":  # This was used in the paper
        diff = probs.unsqueeze(1) - probs.unsqueeze(2)
        disagreement = diff.abs().mean([-3, -2, -1])
    elif mode == "kl":
        marginal_p = probs.mean(dim=0)  # H, D
        marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
        marginal_p = rearrange(marginal_p, "h g d e -> 1 (h g) (d e)")  # 1, H^2, D^2

        pointwise_p = torch.einsum("bhd,bge->bhgde", probs, probs)  # B, H, H, D, D
        pointwise_p = rearrange(
            pointwise_p, "b h g d e -> b (h g) (d e)"
        )  # B, H^2, D^2

        kl_computed = pointwise_p * (pointwise_p.log() - marginal_p.log())
        kl_grid = rearrange(kl_computed.sum(-1), "b (h g) -> b h g", h=heads)
        disagreement = torch.triu(kl_grid, diagonal=1).sum([-1, -2])
    return disagreement.argsort(descending=True)


class DivDisLoss(nn.Module):
    """Computes pairwise repulsion losses for DivDis.

    Args:
        logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * DIM].
        heads (int): Number of heads.
        mode (str): DIVE loss mode. One of {pair_mi, total_correlation, pair_l1}.
    """

    def __init__(self, heads, mode="mi", reduction="mean"):
        super().__init__()
        self.heads = heads
        self.mode = mode
        self.reduction = reduction

    def forward(self, logits):
        heads, mode, reduction = self.heads, self.mode, self.reduction
        probs = to_probs(logits, heads)

        if mode == "mi":  # This was used in the paper
            marginal_p = probs.mean(dim=0)  # H, D
            marginal_p = torch.einsum(
                "hd,ge->hgde", marginal_p, marginal_p
            )  # H, H, D, D
            marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2

            joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(
                dim=0
            )  # H, H, D, D
            joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2

            # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
            # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
            kl_computed = joint_p * (joint_p.log() - marginal_p.log())
            kl_computed = kl_computed.sum(dim=-1)
            kl_grid = rearrange(kl_computed, "(h g) -> h g", h=heads)
            repulsion_grid = -kl_grid
        elif mode == "l1":
            dists = (probs.unsqueeze(1) - probs.unsqueeze(2)).abs()
            dists = dists.sum(dim=-1).mean(dim=0)
            repulsion_grid = dists
        else:
            raise ValueError(f"{mode=} not implemented!")

        if reduction == "mean":  # This was used in the paper
            repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
            repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
            repulsion_loss = -repulsions.mean()
        elif reduction == "min_each":
            repulsion_grid = torch.triu(repulsion_grid, diagonal=1) + torch.tril(
                repulsion_grid, diagonal=-1
            )
            rows = [r for r in repulsion_grid]
            row_mins = [row[row.nonzero(as_tuple=True)].min() for row in rows]
            repulsion_loss = -torch.stack(row_mins).mean()
        else:
            raise ValueError(f"{reduction=} not implemented!")

        return repulsion_loss
