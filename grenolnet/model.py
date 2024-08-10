"""Module to define neural network, which is our solution."""

import math

import torch
from torch import Tensor
from torch.nn import GELU, BatchNorm1d, Linear, Module, ReLU, Sequential, Sigmoid
from torch_geometric.data import Data as PygData
from torch_geometric.nn import NNConv
from torch_geometric.nn import Sequential as PygSequential


class SinusoidalPositionEmbeddings(Module):
    """Class to define network that learns the timestep embedding."""

    def __init__(self, dim: int, device: str | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb_vals = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -emb_vals)
        """Forward pass for the input timestep."""
        embeddings_c = time[:, None] * embeddings[None, :]
        if self.dim % 2 == 1:
            embeddings = torch.cat((embeddings, torch.tensor([1.0], device=device)))
        embeddings_s = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings_s.sin(), embeddings_c.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class GrenolNet(Module):
    """
    Graph convolutional network that predicts the noise applied on the graph nodes.

    Graph Residual Noise Learner Network (GReNoL-Net)
    """

    def __init__(
        self,
        in_views: int,
        out_views: int,
        n_nodes: int,
        conv_size: int,
        time_embeddings: int = 10,
    ) -> None:
        super().__init__()
        self.in_views = in_views
        self.out_views = out_views
        self.n_nodes = n_nodes
        self.conv_size = conv_size
        self.time_embeddings = time_embeddings

        # input-output definitions for PygSequential parameters
        g_in = "x, edge_index, edge_attr"
        g_in_out = "x, edge_index, edge_attr -> x"

        # ----------------------------------------------------------------------------
        # Noise-to-graph generation layers
        # ----------------------------------------------------------------------------

        nn = Sequential(Linear(self.in_views, self.in_views * self.conv_size), ReLU())
        self.source_conv1 = NNConv(self.in_views, self.conv_size, nn, aggr="mean")

        nn = Sequential(Linear(self.in_views, self.conv_size * self.conv_size), ReLU())
        self.source_conv2 = NNConv(self.conv_size, self.conv_size, nn, aggr="mean")

        nn = Sequential(Linear(self.in_views, self.conv_size * self.out_views), ReLU())
        self.source_conv3 = NNConv(self.conv_size, self.out_views, nn, aggr="mean")

        self.time_learner = Sequential(
            SinusoidalPositionEmbeddings(self.n_nodes),
            Linear(self.n_nodes, self.n_nodes),
            GELU(),
            Linear(self.n_nodes, self.n_nodes),
        )

        self.gconv_source_learner = PygSequential(
            g_in,
            [
                (self.source_conv1, g_in_out),
                (self.source_conv2, g_in_out),
                (self.source_conv3, g_in_out),
            ],
        )

        self.fc_mapping = Sequential(
            Linear(self.in_views, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, self.in_views),
            Sigmoid(),
        )

        self.batchnorm_noise_learner = BatchNorm1d(self.n_nodes)

    def forward(
        self, noisy_graph: PygData, source_graph: PygData, timesteps: Tensor
    ) -> Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        noisy_graph: PyG Data object
            A graph with a noise level that the model is expected to predict its noise.
            Expected shape:
            - x: (batch_size, n_nodes, n_features)
            - edge_index: (2, n_nodes * n_nodes * batch_size)
            - edge_attr: (batch_size, n_nodes * n_nodes, n_features)

        source_graph: PyG Data object
            A graph from the source domain that the model is expected to map the information to target domain.
            Expected shape: same as noisy_graph.
        timesteps: Tensor
            Indicates the noise level supposedly added to the noisy_graph during the diffusion process.
            Expected shape: (batch_size)

        Returns
        -------
        torch.Tensor: Predicted noise with shape (batch_size, n_nodes, n_features).
        """
        source_embed = self.gconv_source_learner(
            x=source_graph.x.flatten(end_dim=-2),
            edge_index=source_graph.edge_index,
            edge_attr=source_graph.edge_attr.flatten(end_dim=-2),
        )
        timestep_embed = self.time_learner(timesteps).unsqueeze(-1)

        source_mapped = self.fc_mapping(source_embed)
        source_mapped = source_mapped.reshape_as(noisy_graph.x)

        mapped_features = source_mapped * timestep_embed
        return self.batchnorm_noise_learner(noisy_graph.x) - mapped_features
