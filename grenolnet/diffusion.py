"""Module to handle diffusion processes, mainly adding and removing noise."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as f
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData


class DiffusionBase:
    """Base diffuser common functionalities with conventional diffusion."""

    def __init__(
        self,
        step_count: int = 100,
        noise_schedule: str = "cosine",
        noise_dist: str = "uniform",
        noise_mean: float = 0.5,
        noise_std: float = 0.01,
        noise_max: float = 0.01,
        device: str | None = None,
    ) -> None:
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.step_count = step_count
        self.noise_schedule = noise_schedule
        self.noise_dist = noise_dist
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_max = noise_max
        # Store the in-between graphs generated during the diffusion process.
        self.diff_graphs: list[PygData] = []
        self.edge_index: Tensor | None = None
        self.precalculate_diffusion_params()

    def precalculate_diffusion_params(self) -> None:
        """Calculate diffusion parameters needed for the noise schedule."""
        if self.noise_schedule == "cosine":
            self.betas = self.cosine_beta_schedule(self.step_count).to(self.device)
        elif self.noise_schedule == "quadratic":
            self.betas = self.quadratic_beta_schedule(self.step_count).to(self.device)
        elif self.noise_schedule == "sigmoid":
            self.betas = self.sigmoid_beta_schedule(self.step_count).to(self.device)
        elif self.noise_schedule == "linear":
            self.betas = self.linear_beta_schedule(self.step_count).to(self.device)
        else:
            raise NotImplementedError(
                "Specified noise schedule is not implemented yet."
                + " Available ones are: cosine and linear."
            )
        # Pre-calculate different terms for closed form
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = f.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def linear_beta_schedule(
        timesteps: int, start: float = 0.0001, end: float = 0.02
    ) -> Tensor:
        """Apply default linear schedule."""
        return torch.linspace(start, end, timesteps)

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
        """Apply cosine schedule as proposed in https://arxiv.org/abs/2102.09672."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # noqa: PLR6104
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def quadratic_beta_schedule(
        timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> Tensor:
        """Apply quadratic schedule."""
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(
        timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> Tensor:
        """Apply sigmoid schedule."""
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def create_noise_like(self, tensor: Tensor) -> Tensor:
        """Create a noise tensor with the given size."""
        if self.noise_dist == "uniform":
            noise = (
                torch.rand(tensor.shape, device=self.device, requires_grad=True)
                * self.noise_max
            )
        elif self.noise_dist == "normal":
            noise = torch.normal(
                self.noise_mean,
                self.noise_std,
                size=tensor.shape,
                requires_grad=True,
                device=self.device,
            )
        else:
            noise = torch.randn_like(tensor, requires_grad=True, device=self.device)
        return noise

    def q_sample(
        self,
        tensor: Tensor,
        timestep: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Add noise to the input tensor."""
        if noise is None:
            tensor = tensor.unsqueeze(-1)
            noise = self.create_noise_like(tensor)
        noise_mean = self.sqrt_alphas_cumprod
        noise_var = self.sqrt_one_minus_alphas_cumprod
        if timestep is None:
            noisy_tensor = noise_mean * tensor + noise_var * noise
        else:
            noisy_tensor = (
                noise_mean[timestep].view(-1, 1, 1) * tensor
                + noise_var[timestep].view(-1, 1, 1) * noise
            )
        return noisy_tensor, noise.squeeze(-1)

    @torch.no_grad()
    def p_sample(
        self,
        model_input: Tensor,
        model_output: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Remove noise from a noisy input tensor."""
        noise_mean = self.sqrt_recip_alphas[timesteps]
        noise_var = self.sqrt_one_minus_alphas_cumprod[timesteps]
        model_mean = noise_mean * (
            model_input - self.betas[timesteps] * model_output / noise_var
        )
        if timesteps[0] == 0:
            return model_mean
        return model_mean + torch.sqrt(self.posterior_variance[timesteps]) * noise


class GraphDiffusion(DiffusionBase):
    """Diffuser to handle graph-specific diffusion processes."""

    def __init__(
        self,
        step_count: int = 100,
        noise_schedule: str = "cosine",
        noise_dist: str = "uniform",
        noise_mean: float = 0.5,
        noise_std: float = 0.01,
        noise_max: float = 0.01,
        device: str | None = None,
    ) -> None:
        super().__init__(
            step_count,
            noise_schedule,
            noise_dist,
            noise_mean,
            noise_std,
            noise_max,
            device,
        )

    @staticmethod
    def calculate_edges(nodes: Tensor) -> Tensor:
        """Calculate edges based on the node features and pairwise function."""
        edges = torch.abs(nodes.unsqueeze(1) - nodes) / (nodes.unsqueeze(1) + nodes)
        return edges.flatten(end_dim=-2)

    @staticmethod
    def create_edge_connectivity(n_nodes: int, device: str | None = None) -> Tensor:
        """Prepare a COO matrix to pair nodes."""
        # Torch operations execute faster to create source-destination pairs.
        # [0,1,2,3,0,1,2,3...]
        dst_index = torch.arange(n_nodes, device=device).repeat(n_nodes)
        # [0,0,0,0,1,1,1,1...]
        src_index = (
            torch.arange(n_nodes, device=device)
            .expand(n_nodes, n_nodes)
            .transpose(0, 1)
            .reshape(n_nodes * n_nodes)
        )
        # COO Matrix for index src-dst pairs. And add batch dimensions.
        return torch.stack([src_index, dst_index])

    def q_sample_graph(self, nodes: Tensor, timestep: Tensor, noise: Tensor) -> PygData:
        """Add specified noise on specified level and return result as graph."""
        noisy_nodes_batch, _ = self.q_sample(nodes, timestep, noise)
        _, n_nodes, _ = noisy_nodes_batch.shape

        if self.edge_index is None:
            self.edge_index = self.create_edge_connectivity(n_nodes, device=self.device)

        noisy_graph_list = []
        noisy_graph_batch = PygBatch()
        for noisy_nodes in noisy_nodes_batch:
            noisy_edge_attr = self.calculate_edges(noisy_nodes)
            noisy_graph = PygData(
                x=noisy_nodes.unsqueeze(0),
                edge_index=self.edge_index,
                edge_attr=noisy_edge_attr.unsqueeze(0),
            )
            noisy_graph_list.append(noisy_graph)
        return noisy_graph_batch.from_data_list(noisy_graph_list)

    def forward_add_noise(
        self, node_feature_matrix: Tensor
    ) -> tuple[list[PygData], Tensor]:
        """Add noise and return results from all steps."""
        noisy_nodes_all, noise = self.q_sample(node_feature_matrix)
        noisy_nodes_all = noisy_nodes_all.permute(3, 0, 1, 2)
        noisy_nodes_all = torch.clamp(noisy_nodes_all, 0.0, 1.0)
        _, _, n_nodes, _ = noisy_nodes_all.shape
        if self.edge_index is None:
            self.edge_index = self.create_edge_connectivity(n_nodes, device=self.device)
        noisy_graphs = []
        # Create diffused graphs as much as diff_sample_size
        for noisy_nodes_level in noisy_nodes_all:
            noisy_edge_attr = self.calculate_edges(noisy_nodes_level.squeeze(0))
            noisy_graph = PygData(
                x=noisy_nodes_level,
                edge_index=self.edge_index,
                edge_attr=noisy_edge_attr.unsqueeze(0),
            )
            noisy_graphs.append(noisy_graph)
        return noisy_graphs, noise

    def forward_add_noise_to_batch(
        self, graph_batch: PygData
    ) -> tuple[PygData, Tensor]:
        """Add noise and return results from all steps."""
        # Create diffused graphs as much as diff_sample_size
        noisy_nodes_all, noise = self.q_sample(graph_batch.x)
        noisy_nodes_all = noisy_nodes_all.permute(3, 0, 1, 2)
        noisy_nodes_all = torch.clamp(noisy_nodes_all, 0.0, 1.0)
        _, _, n_nodes, _ = noisy_nodes_all.shape
        if self.edge_index is None:
            self.edge_index = self.create_edge_connectivity(n_nodes, device=self.device)
        # Create diffused graphs as much as diff_sample_size
        noisy_graphs = []
        for noisy_nodes in noisy_nodes_all:
            noisy_batch_list = []
            for batch_sample in noisy_nodes:
                noisy_edge_attr = self.calculate_edges(batch_sample)
                noisy_graph = PygData(
                    x=batch_sample.unsqueeze(0),
                    edge_index=self.edge_index,
                    edge_attr=noisy_edge_attr.unsqueeze(0),
                )
                noisy_batch_list.append(noisy_graph)
            noisy_batch = PygBatch().from_data_list(noisy_batch_list)
            noisy_graphs.append(noisy_batch)
        return noisy_graphs, noise

    @torch.no_grad()
    def forward_remove_noise(
        self,
        model_or_outputs: Tensor | Module,
        noisy_graph: PygData,
        source_graph: PygData | None = None,
        noise: Tensor | None = None,
    ) -> list[PygData]:
        """Recursively remove noise from the noisy input based on the provided model output."""
        if noise is None:
            noise = self.create_noise_like(noisy_graph.x)
        _, n_nodes, _ = noise.shape

        if self.edge_index is None:
            self.edge_index = self.create_edge_connectivity(n_nodes, device=self.device)

        denoised_graphs = []
        denoised_graph = noisy_graph
        batch_size, _, _ = noisy_graph.x.shape
        for t_idx in range(self.step_count)[::-1]:
            timesteps = torch.full(
                (batch_size,), t_idx, device=self.device, dtype=torch.long
            )

            if isinstance(model_or_outputs, Module):
                pred_noise = model_or_outputs(denoised_graph, source_graph, timesteps)
                # pred_noise = pred_noise.unsqueeze(0)
            elif isinstance(model_or_outputs, Tensor):
                pred_noise = torch.clone(model_or_outputs)
            else:
                raise TypeError(
                    "model_or_output is not a list of graph outputs nor a Module"
                )

            denoised_node_matrix = self.p_sample(
                denoised_graph.x, pred_noise, noise, timesteps
            )
            # denoised_node_matrix = torch.clamp(denoised_node_matrix, 0.0, 1.0)

            denoised_batch_list = []
            for denoised_nodes in denoised_node_matrix:
                denoised_edges = self.calculate_edges(denoised_nodes)
                denoised_graph = PygData(
                    x=denoised_nodes.unsqueeze(0),
                    edge_index=self.edge_index,
                    edge_attr=denoised_edges.unsqueeze(0),
                )
                denoised_batch_list.append(denoised_graph)
            denoised_batch = PygBatch().from_data_list(denoised_batch_list)
            denoised_graphs.append(denoised_batch)

        return denoised_graphs
