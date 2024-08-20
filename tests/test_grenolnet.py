"""Test graph dataset classes."""

from __future__ import annotations

import os
from sys import platform

import torch
from torch_geometric.data import Data as PygData

from grenolnet.diffusion import GraphDiffusion
from grenolnet.inference import BaseInferer
from grenolnet.model import GrenolNet
from grenolnet.train import BaseTrainer

DATA_PATH = os.path.join(os.path.dirname(__file__), "mock_datasets")
GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "expected")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
DEVICE = "cpu"


def test_simple_iteration() -> None:
    """Test if the model can be iterated - cpu based."""
    n_subjects = 2
    n_nodes = 34
    xs = torch.rand(n_subjects, n_nodes, 1)
    edge_attr = torch.rand(n_subjects, n_nodes * n_nodes, 1)
    edge_index = GraphDiffusion.create_edge_connectivity(n_nodes)
    edge_index = torch.stack([edge_index, edge_index], dim=-1).flatten(start_dim=-2)
    noisy_graph = PygData(x=xs, edge_index=edge_index, edge_attr=edge_attr)
    source_graph = PygData(x=xs, edge_index=edge_index, edge_attr=edge_attr)
    timesteps = torch.randint(0, 1000, (n_subjects,)).long()

    grenolnet = GrenolNet(in_views=1, out_views=1, n_nodes=n_nodes, conv_size=50)
    out = grenolnet.forward(noisy_graph, source_graph, timesteps)
    assert out.shape == (n_subjects, n_nodes, 1)


def test_reproducibility() -> None:
    """Test if the model can give same results always - compare cpu based results with cuda results."""
    n_subjects = 2
    n_nodes = 34
    xs = torch.rand(n_subjects, n_nodes, 1)
    edge_attr = torch.rand(n_subjects, n_nodes * n_nodes, 1)
    edge_index = GraphDiffusion.create_edge_connectivity(n_nodes)
    edge_index = torch.stack([edge_index, edge_index], dim=-1).flatten(start_dim=-2)
    noisy_graph = PygData(x=xs, edge_index=edge_index, edge_attr=edge_attr)
    source_graph = PygData(x=xs, edge_index=edge_index, edge_attr=edge_attr)
    timesteps = torch.randint(0, 1000, (n_subjects,)).long()

    grenolnet_eval = GrenolNet(in_views=1, out_views=1, n_nodes=n_nodes, conv_size=50)
    grenolnet_eval.eval()
    out1 = grenolnet_eval.forward(noisy_graph, source_graph, timesteps)
    out2 = grenolnet_eval.forward(noisy_graph, source_graph, timesteps)
    if platform != "win32":
        # Somehow windows system does not give stable results.
        assert torch.equal(out1, out2)


def test_diffusion_add_noise() -> None:
    """Test if the diffusing graphs works properly - cpu based."""
    n_subjects = 2
    n_nodes = 25

    xs = torch.rand(n_subjects, n_nodes, 1)
    edge_attr = torch.rand(n_subjects, n_nodes * n_nodes, 1)
    edge_index = GraphDiffusion.create_edge_connectivity(n_nodes)
    edge_index = torch.stack([edge_index] * n_subjects, dim=-1).flatten(start_dim=-2)
    source_graph = PygData(x=xs, edge_index=edge_index, edge_attr=edge_attr)

    diffuser = GraphDiffusion(device="cpu")
    noisy_graphs, noise = diffuser.forward_add_noise_to_batch(source_graph)
    assert noise.shape == (n_subjects, n_nodes, 1)
    assert noisy_graphs[-1].x.shape == (n_subjects, n_nodes, 1)
    assert noisy_graphs[-1].edge_attr.shape == (n_subjects, n_nodes * n_nodes, 1)
    assert noisy_graphs[-1].edge_index.shape == (2, n_nodes * n_nodes * n_subjects)


def test_diffusion_remove_noise() -> None:
    """Test if the diffusing graphs works properly - cpu based."""
    n_subjects = 1
    n_nodes = 35

    xs = torch.rand(n_subjects, n_nodes, 1)
    edge_attr = torch.rand(n_subjects, n_nodes * n_nodes, 1)
    edge_index = GraphDiffusion.create_edge_connectivity(n_nodes)
    edge_index = torch.stack([edge_index] * n_subjects, dim=-1).flatten(start_dim=-2)
    source_graph = PygData(x=xs, edge_index=edge_index, edge_attr=edge_attr)

    diffuser = GraphDiffusion(device="cpu")
    noise = diffuser.create_noise_like(source_graph.x)
    noisy_edge_attr = diffuser.calculate_edges(noise.squeeze(0))
    edge_index = diffuser.create_edge_connectivity(len(noise.squeeze()))
    full_noise_graph = PygData(
        x=noise,
        edge_index=edge_index,
        edge_attr=noisy_edge_attr.unsqueeze(0),
    )

    grenolnet = GrenolNet(in_views=1, out_views=1, n_nodes=n_nodes, conv_size=50)
    grenolnet.eval()

    denoised_graphs = diffuser.forward_remove_noise(
        grenolnet, full_noise_graph, source_graph
    )
    assert len(denoised_graphs) == 100
    assert denoised_graphs[-1].x.shape == (n_subjects, n_nodes, 1)
    assert denoised_graphs[-1].edge_attr.shape == (n_subjects, n_nodes * n_nodes, 1)
    assert denoised_graphs[-1].edge_index.shape == (2, n_nodes * n_nodes * n_subjects)


def test_trainer() -> None:
    """Test if the experiment module works properly."""
    trainer = BaseTrainer(
        hemisphere="left",
        dataset="openneuro",
        timepoint=None,
        n_epochs=3,
        learning_rate=0.001,
        validation_period=1,
        diffusion_step_count=10,
        loss_weight=0.0,
        loss_name="mse",
        batch_size=2,
        random_seed=0,
        device=DEVICE,
    )
    trainer.train()


def test_inferer() -> None:
    """Test if the experiment module works properly."""
    diff_params = {
        "step_count": 10,
        "noise_schedule": "cosine",
        "noise_dist": "normal",
        "noise_mean": 0.5,
        "noise_std": 0.01,
        "noise_max": 0.02,
    }
    model_params = {
        "conv_size": 48,
        "in_views": 1,
        "out_views": 1,
        "n_nodes": 34,
    }
    target_model_path = os.path.join(
        MODELS_PATH,
        "op_bl_left_500epochs_5folds_batchnorm",
        "op_bl_left_500epochs_5folds_batchnorm_fold0.pth",
    )
    inferer = BaseInferer(
        dataset="openneuro",
        hemisphere="left",
        src_feature_idx=0,
        tgt_feature_idx=2,
        model_path=target_model_path,
        model_params=model_params,
        diffusion_params=diff_params,
        random_seed=0,
        device=DEVICE,
    )
    current_results = inferer.run("test")
    assert "frobenius" in current_results
    assert "edges_mse" in current_results
    cuda_results = torch.load(
        os.path.join(GOLD_STANDARD_PATH, "subject_wise_frobenius_results_cuda.pth"),
        map_location=torch.device("cpu"),
    )
    cpu_results = torch.load(
        os.path.join(GOLD_STANDARD_PATH, "subject_wise_frobenius_results_cpu.pth"),
    )
    assert cuda_results is not None
    assert cuda_results.get_device() == -1  # -1 is cpu
    if platform != "win32":
        # Somehow windows system does not give stable results.
        assert torch.isclose(current_results["frobenius"], cpu_results).all()
    # assert torch.isclose(results["frobenius"], cuda_results).all()
