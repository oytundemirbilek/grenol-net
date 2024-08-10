"""Module for training script."""

from __future__ import annotations

import json
import locale
import os

import numpy as np
import torch
from avicortex.datasets import (
    CandiShareSchizophreniaDataset,
    HCPYoungAdultDataset,
    OpenNeuroCannabisUsersDataset,
)
from torch.nn import Module
from torch.nn.modules.loss import MSELoss, _Loss
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from grenolnet.diffusion import GraphDiffusion
from grenolnet.evaluation import FrobeniusDistance, JointMSELoss
from grenolnet.model import GrenolNet

# These two options should be seed to ensure reproducible (If you are using cudnn backend)
# https://pytorch.org/docs/stable/notes/randomness.html
if torch.cuda.is_available():
    from torch.backends import cudnn

    cudnn.deterministic = True
    cudnn.benchmark = False

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


class BaseTrainer:
    """Wrapper around training function to save all the training parameters."""

    def __init__(
        self,
        # Data related:
        hemisphere: str,
        dataset: str,
        timepoint: str | None,
        # Training related:
        n_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.001,
        batch_size: int = 1,
        validation_period: int = 5,
        modelsaving_period: int = 5,
        patience: int | None = None,
        src_feature_idx: int = 0,
        tgt_feature_idx: int = 2,
        # Model related:
        n_folds: int = 5,
        loss_weight: float = 1.0,
        loss_name: str = "joint_mse",
        conv_size: int = 48,
        model_name: str = "default_model_name",
        # diffusion related:
        diffusion_step_count: int = 1000,
        diffusion_noise_schedule: str = "cosine",
        diffusion_noise_dist: str = "uniform",
        diffusion_noise_mean: float = 0.5,
        diffusion_noise_std: float = 0.1,
        diffusion_noise_max: float = 0.1,
        random_seed: int = 0,
        device: str | None = None,
    ) -> None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.random_seed = random_seed
        self.hemisphere = hemisphere
        self.dataset = dataset
        self.timepoint = timepoint
        self.n_epochs = n_epochs
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.src_feature_idx = src_feature_idx
        self.tgt_feature_idx = tgt_feature_idx
        self.validation_period = validation_period
        self.modelsaving_period = modelsaving_period
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.diffusion_step_count = diffusion_step_count
        self.diffusion_noise_schedule = diffusion_noise_schedule
        self.diffusion_noise_dist = diffusion_noise_dist
        self.diffusion_noise_mean = diffusion_noise_mean
        self.diffusion_noise_std = diffusion_noise_std
        self.diffusion_noise_max = diffusion_noise_max
        self.conv_size = conv_size
        self.model_name = model_name
        self.model_save_path = os.path.join(FILE_PATH, "..", "models", model_name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.model_save_path = os.path.join(self.model_save_path, model_name)

        self.model_params_save_path = os.path.join(
            FILE_PATH, "..", "models", model_name + "_params.json"
        )
        with open(
            self.model_params_save_path,
            "w",
            encoding=locale.getpreferredencoding(False),
        ) as f:
            json.dump(self.__dict__, f, indent=4)

        self.loss_fn: _Loss
        if loss_name == "joint_mse":
            self.loss_fn = JointMSELoss()
        elif loss_name == "mse":
            self.loss_fn = MSELoss()
        elif loss_name == "frobenius":
            self.loss_fn = FrobeniusDistance()
        else:
            raise NotImplementedError(
                "Pick one of the losses: 'joint_mse' or 'frobenius'"
            )

        self.val_loss_per_epoch: list[float] = []

    def __repr__(self) -> str:
        """Return string representation of the Trainer as training parameters."""
        return str(self.__dict__)

    @torch.no_grad()
    def validate(
        self,
        model: Module,
        val_dataloader: DataLoader,
    ) -> float:
        """Run validation loop."""
        model.eval()
        val_diffuser = GraphDiffusion(
            step_count=self.diffusion_step_count,
            noise_schedule=self.diffusion_noise_schedule,
            noise_dist=self.diffusion_noise_dist,
            noise_mean=self.diffusion_noise_mean,
            noise_std=self.diffusion_noise_std,
            noise_max=self.diffusion_noise_max,
            device=self.device,
        )

        all_val_losses = []
        for source_graph, target_graph in val_dataloader:
            timestep = torch.randint(
                0,
                val_diffuser.step_count,
                (target_graph.x.shape[0],),
                device=self.device,
            ).long()
            applied_noise = val_diffuser.create_noise_like(target_graph.x)
            noisy_graph = val_diffuser.q_sample_graph(
                target_graph.x, timestep, applied_noise
            )
            pred_noise = model.forward(noisy_graph, source_graph, timestep)
            val_loss = self.loss_fn(pred_noise, applied_noise.reshape_as(pred_noise))
            all_val_losses.append(val_loss)

        model.train()
        return torch.stack(all_val_losses).mean().item()

    def train(self, current_fold: int = 0) -> Module:
        """Train model."""
        tr_dataset = DATASETS[self.dataset](
            hemisphere=self.hemisphere,
            freesurfer_out_path=DATASET_PATHS[self.dataset],
            mode="train",
            n_folds=self.n_folds,
            current_fold=current_fold,
            src_view_idx=self.src_feature_idx,
            tgt_view_idx=self.tgt_feature_idx,
            random_seed=self.random_seed,
            device=self.device,
        )
        val_dataset = DATASETS[self.dataset](
            hemisphere=self.hemisphere,
            freesurfer_out_path=DATASET_PATHS[self.dataset],
            mode="validation",
            n_folds=self.n_folds,
            current_fold=current_fold,
            src_view_idx=self.src_feature_idx,
            tgt_view_idx=self.tgt_feature_idx,
            random_seed=self.random_seed,
            device=self.device,
        )
        print("Training data: ", tr_dataset)
        print("Validation data: ", val_dataset)
        tr_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        diffuser = GraphDiffusion(
            step_count=self.diffusion_step_count,
            noise_schedule=self.diffusion_noise_schedule,
            noise_dist=self.diffusion_noise_dist,
            noise_mean=self.diffusion_noise_mean,
            noise_std=self.diffusion_noise_std,
            noise_max=self.diffusion_noise_max,
            device=self.device,
        )
        model = GrenolNet(
            in_views=1, out_views=1, n_nodes=34, conv_size=self.conv_size
        ).to(self.device)

        optimizer = AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.best_val_loss = 9999999999.0
        self.best_model = model
        for epoch in range(self.n_epochs):
            all_tr_losses = []
            for source_graph, target_graph in tr_dataloader:
                timestep = torch.randint(
                    0,
                    diffuser.step_count,
                    (target_graph.x.shape[0],),
                    device=self.device,
                ).long()
                applied_noise = diffuser.create_noise_like(target_graph.x)
                noisy_graph = diffuser.q_sample_graph(
                    target_graph.x, timestep, applied_noise
                )
                pred_noise = model.forward(noisy_graph, source_graph, timestep)
                tr_loss = self.loss_fn(pred_noise, applied_noise.reshape_as(pred_noise))

                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

                all_tr_losses.append(tr_loss.detach())

            avg_tr_loss = torch.stack(all_tr_losses).mean().item()

            if (epoch + 1) % self.validation_period == 0:
                val_loss = self.validate(model, val_dataloader)
                print(
                    f"Epoch: {epoch + 1}/{self.n_epochs} | Tr.Loss: {avg_tr_loss} | Val.Loss: {val_loss}"
                )
                self.val_loss_per_epoch.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = model

                    torch.save(
                        model.state_dict(),
                        self.model_save_path + f"_fold{current_fold}.pth",
                    )

        return self.best_model
