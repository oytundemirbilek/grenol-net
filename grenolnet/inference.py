"""Module for inference and testing scripts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from avicortex.datasets import (
    CandiShareSchizophreniaDataset,
    GraphDataset,
    HCPYoungAdultDataset,
    OpenNeuroCannabisUsersDataset,
)
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data as PygData
from torch_geometric.loader import DataLoader

from grenolnet.diffusion import GraphDiffusion
from grenolnet.evaluation import FrobeniusDistance, JointMSELoss
from grenolnet.model import GrenolNet

FILE_PATH = os.path.dirname(__file__)

DATASETS = {
    "candishare": CandiShareSchizophreniaDataset,
    "openneuro": OpenNeuroCannabisUsersDataset,
    "hcp": HCPYoungAdultDataset,
}
DATASET_PATHS = {
    "candishare": os.path.join(
        FILE_PATH, "..", "datasets", "candishare_schizophrenia_dktatlas.csv"
    ),
    "openneuro": os.path.join(
        FILE_PATH, "..", "datasets", "openneuro_baseline_dktatlas.csv"
    ),
    "hcp": os.path.join(FILE_PATH, "..", "datasets", "hcp_young_adult.csv"),
}


class BaseInferer:
    """Inference loop for a trained model. Run the testing schema."""

    def __init__(
        self,
        dataset: str,
        hemisphere: str,
        diffusion_params: dict[str, Any],
        src_feature_idx: int = 0,
        tgt_feature_idx: int = 2,
        model: Module | None = None,
        model_path: str | None = None,
        model_params: dict[str, Any] | None = None,
        random_seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.model_path = model_path
        self.random_seed = random_seed
        self.model_params = model_params
        self.model: Module
        self.diffusion_params = diffusion_params
        self.src_feature_idx = src_feature_idx
        self.tgt_feature_idx = tgt_feature_idx

        if model is None:
            if model_params is None:
                raise ValueError("Specify a model or model params and its path.")
            if model_path is None:
                raise ValueError("Specify a model or model params and its path.")
            self.model = self.load_model_from_file(
                model_path, model_params, self.device
            )
        else:
            self.model = model

        self.dataset = dataset
        self.hemisphere = hemisphere

        self.metric_frobenius = FrobeniusDistance()
        self.metric_joint_mse = JointMSELoss()

    @torch.no_grad()
    def run(
        self, mode: str = "test", save_predictions: bool = False
    ) -> dict[str, Tensor]:
        """Infer the models after the training is finished, and collect test results."""
        self.model.eval()

        test_dataset: GraphDataset = DATASETS[self.dataset](
            hemisphere=self.hemisphere,
            freesurfer_out_path=DATASET_PATHS[self.dataset],
            mode=mode,
            n_folds=1,
            src_view_idx=self.src_feature_idx,
            tgt_view_idx=self.tgt_feature_idx,
            random_seed=self.random_seed,
            device=self.device,
        )
        self.diffusion_params["device"] = self.device
        print("Testing data: ", test_dataset)
        test_diffuser = GraphDiffusion(**self.diffusion_params)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        test_losses_per_sample: dict[str, list[float]] = {
            "frobenius": [],
            "edges_mse": [],
            "nodes_mse": [],
            "joint_mse": [],
        }

        # target_stats = np.load("./cs_left_tr_target_stats.npy", allow_pickle=False)
        # target_stats = torch.from_numpy(target_stats).to(device)
        save_path = os.path.join(
            FILE_PATH,
            "..",
            "benchmarks",
            "grenol-net",
            f"{self.dataset}_{self.hemisphere}",
        )
        if not os.path.exists(os.path.join(save_path, "predictions")):
            os.makedirs(os.path.join(save_path, "predictions"))
        if not os.path.exists(os.path.join(save_path, "targets")):
            os.makedirs(os.path.join(save_path, "targets"))
        for s_idx, (source_graph, target_graph) in enumerate(test_dataloader):
            # normal dist: mu = 0.5, sig = 0.1 -> frob =~ 3.75, mse =~ 0.0125
            # normal dist: mu = 0.5, sig = 0.01 -> frob =~ 2.82, mse =~ 0.0070
            # uniform dist: from = 0.0, to = 0.1 -> doesn't work
            # uniform dist x 10: from = 0.0, to = 0.1 -> doesn't work
            # all 0.5 dist -> frob =~ 3.10, mse =~ 0.0085
            noise = test_diffuser.create_noise_like(source_graph.x)
            noisy_edge_attr = test_diffuser.calculate_edges(noise.squeeze(0))
            edge_index = test_diffuser.create_edge_connectivity(
                len(noise.squeeze()), device=self.device
            )
            full_noise_graph = PygData(
                x=noise,
                edge_index=edge_index,
                edge_attr=noisy_edge_attr.unsqueeze(0),
            )
            # noisy_graphs, applied_noise = test_diffuser.forward_add_noise(gaussian1)
            denoised_graphs = test_diffuser.forward_remove_noise(
                self.model,
                full_noise_graph,
                source_graph=source_graph,
            )

            # When the noise is fully removed, check how closer it is to target graph.
            test_loss_frobenius = self.metric_frobenius(
                denoised_graphs[-1], target_graph
            )
            test_losses_per_sample["frobenius"].append(test_loss_frobenius.item())
            test_loss_joint_mse = self.metric_joint_mse(
                denoised_graphs[-1], target_graph
            )
            test_losses_per_sample["edges_mse"].append(
                self.metric_joint_mse.edges_loss.item()
            )
            test_losses_per_sample["nodes_mse"].append(
                self.metric_joint_mse.nodes_loss.item()
            )
            test_losses_per_sample["joint_mse"].append(test_loss_joint_mse.item())
            # titles = reverse_indices.tolist()[::10]
            # plot_graph_adjacency(denoised_graphs[::10], titles)
            # plot_graph_adjacency([source_graph, target_graph], ["source", "target"])
            if save_predictions:
                adj_matrix_pred = (
                    denoised_graphs[-1].edge_attr.reshape(34, 34).cpu().numpy()
                )
                adj_matrix_tgt = target_graph.edge_attr.reshape(34, 34).cpu().numpy()
                np.savetxt(
                    os.path.join(save_path, "predictions", f"subject{s_idx}_pred.txt"),
                    adj_matrix_pred,
                )
                np.savetxt(
                    os.path.join(save_path, "targets", f"subject{s_idx}_tgt.txt"),
                    adj_matrix_tgt,
                )

        self.model.train()
        return self._test_losses_list_to_tensor(test_losses_per_sample)

    @staticmethod
    def _test_losses_list_to_tensor(
        losses: dict[str, list[float]]
    ) -> dict[str, Tensor]:
        test_losses_tensors: dict[str, Tensor] = {}

        for key, value in losses.items():
            test_losses_tensors[key] = torch.tensor(value)

        return test_losses_tensors

    @staticmethod
    def load_model_from_file(
        model_path: str, model_params: dict[str, Any], device: str | None = None
    ) -> Module:
        """Load a model from the path and based on defined parameters."""
        model = GrenolNet(**model_params).to(device)
        if not model_path.endswith(".pth"):
            model_path += ".pth"
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device) if device is not None else None,
            )
        )
        return model
