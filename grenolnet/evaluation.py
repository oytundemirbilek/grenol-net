"""Module that provides calculations for loss functions and evaluation metrics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from networkx.algorithms import centrality
from torch import Tensor
from torch.nn import MSELoss
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data as PygData
from torch_geometric.utils import to_networkx


class JointMSELoss(MSELoss):
    """Loss that combines MSE losses on edge features and node features on a graph."""

    def __init__(
        self,
        reduction: str = "mean",
        node_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.node_loss_weight = node_loss_weight

    def forward(self, source: PygData, target: PygData) -> Tensor:
        """Calculate MSE loss between input and target graphs."""
        self.edges_loss = super().forward(source.edge_attr, target.edge_attr)
        self.nodes_loss = super().forward(source.x, target.x)
        return self.node_loss_weight * self.nodes_loss + self.edges_loss


class FrobeniusDistance(_Loss):
    """Frobenius distance loss to optimize more topological information for the graphs."""

    def __init__(
        self,
        batch_first: bool = True,
        reduction: str = "mean",
        node_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.node_loss_weight = node_loss_weight
        self.batch_first = batch_first
        self.reduction = reduction

    def forward(
        self, pred_data: PygData | Tensor, target_data: PygData | Tensor
    ) -> Tensor:
        """Calculate Frobenius loss between input and target graphs."""
        if not isinstance(pred_data, Tensor):
            pred_data = pred_data.edge_attr
        if not isinstance(target_data, Tensor):
            target_data = target_data.edge_attr

        if self.batch_first:
            self.loss = torch.sqrt(
                torch.square(pred_data - target_data).sum(dim=(1, 2))
            )
        else:
            self.loss = torch.sqrt(torch.square(pred_data - target_data).sum())
        if self.reduction == "mean":
            return self.loss.mean()
        elif self.reduction == "sum":
            return self.loss.sum()
        else:
            raise NotImplementedError(
                "Batch reduction options are only 'mean' or 'sum'"
            )


def pearson_loss_regions_2(pred: Tensor, target: Tensor) -> Tensor:
    """Pearson correlation coefficient between the prediction and target."""
    eps = 1e-6

    pred_mean = pred - pred.mean(dim=0) + eps
    target_mean = target - target.mean(dim=0) + eps
    print(target_mean)
    r_num = torch.sum(pred_mean * target_mean, dim=0)
    r_den = torch.sqrt(
        torch.sum(torch.pow(pred_mean, 2), dim=0)
        * torch.sum(torch.pow(target_mean, 2), dim=0)
    )
    pear = r_num / r_den
    return torch.pow(pear - 1.0, 2).sum()


def pearson_loss_regions(pred: Tensor, target: Tensor) -> float:
    """Pearson correlation coefficient between the prediction and target."""
    loss = 0.0
    eps = 1e-6
    region_num = pred.shape[0]
    for region in range(region_num):
        pred_mean = pred[region] - torch.mean(pred[region]) + eps
        target_mean = target[region] - torch.mean(target[region]) + eps
        print(target_mean)
        r_num = torch.sum(pred_mean * target_mean)
        r_den = torch.sqrt(
            torch.sum(torch.pow(pred_mean, 2)) * torch.sum(torch.pow(target_mean, 2))
        )
        pear = r_num / r_den
        # pear_official = signal_corelation(gen_vec.numpy(), real_vec.numpy())
        loss = loss + torch.pow(pear - 1.0, 2).item()
    return loss


class Centrality(_Loss):
    """Centrality difference loss to optimize more topological information for the graphs."""

    def __init__(
        self,
        batch_first: bool = True,
        reduction: str = "mean",
        node_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.node_loss_weight = node_loss_weight
        self.batch_first = batch_first
        self.reduction = reduction

    def forward(
        self,
        pred_data: PygData | np.ndarray,
        target_data: PygData | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the difference in closeness centrality between input and target graphs."""
        if isinstance(pred_data, PygData):
            pred_g = to_networkx(pred_data)
        if isinstance(target_data, PygData):
            tgt_g = to_networkx(target_data)

        if isinstance(pred_data, np.ndarray):
            pred_g = nx.from_numpy_array(pred_data)
        if isinstance(target_data, np.ndarray):
            tgt_g = nx.from_numpy_array(target_data)

        pred_centralities = centrality.closeness_centrality(pred_g, distance="weight")
        tgt_centralities = centrality.closeness_centrality(tgt_g, distance="weight")

        return np.fromiter(pred_centralities.values(), dtype=float), np.fromiter(
            tgt_centralities.values(), dtype=float
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(5, 5)
    tgt = torch.rand(5, 5)
    # print(pearson_loss_regions(x, tgt))
    print(pearson_loss_regions_2(x, tgt))
