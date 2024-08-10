"""Module to run benchmarking code to compare results with our model."""

from __future__ import annotations

import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from avicortex.builders import GraphBuilder
from avicortex.datasets import (
    CandiShareSchizophreniaDataset,
    GraphDataset,
    HCPYoungAdultDataset,
    OpenNeuroCannabisUsersDataset,
)
from torch_geometric.loader import DataLoader

from grenolnet.evaluation import Centrality

DATASETS = {
    "candishare": CandiShareSchizophreniaDataset,
    "openneuro": OpenNeuroCannabisUsersDataset,
    "hcp": HCPYoungAdultDataset,
}
FILE_DIR = os.path.dirname(__file__)


class BaseBenchmark:
    """Base class to cover common functionalities across benchmark methods."""

    def __init__(
        self, model_name: str, dataset: str, hemisphere: str, cortical_feature: int = 0
    ) -> None:
        self.cortical_feature = cortical_feature
        self.dataset = dataset
        self.hemisphere = hemisphere
        # self.metric_frobenius = FrobeniusDistance()
        # self.metric_joint_mse = MSELoss()
        # self.metric_discriminativeness = Discriminativeness()
        self.topological_metric = Centrality()
        self.root_dir = os.path.join(FILE_DIR, "..", "benchmarks", model_name)
        self.model_dir = os.path.join(
            self.root_dir, dataset + "_" + hemisphere, "models"
        )
        self.results_dir = os.path.join(self.root_dir, dataset + "_" + hemisphere)

    def __repr__(self) -> str:
        """Return string representation of the class."""
        return f"{self.__class__.__name__} benchmarker - {self.dataset} - {self.hemisphere}"

    @staticmethod
    def collect_data_from_loader(
        dataloader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect source and target graphs from the dataloader and return them as numpy arrays."""
        sources = []
        targets = []
        for source_graph, target_graph in dataloader:
            source_arr = source_graph.x.squeeze().detach().cpu().numpy()
            target_arr = target_graph.x.squeeze().detach().cpu().numpy()
            sources.append(source_arr)
            targets.append(target_arr)
        return np.array(sources), np.array(targets)

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Collect data from training dataloader."""
        tr_dataset: GraphDataset = DATASETS[self.dataset](
            hemisphere=self.hemisphere,
            mode="train",
            n_folds=5,
            src_view_idx=0,
            tgt_view_idx=2,
        )
        tr_dataloader = DataLoader(tr_dataset, batch_size=1)
        return self.collect_data_from_loader(tr_dataloader)

    def get_testing_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Collect data from testing dataloader."""
        test_dataset: GraphDataset = DATASETS[self.dataset](
            hemisphere=self.hemisphere,
            mode="test",
            n_folds=1,
            tgt_view_idx=0,
            src_view_idx=2,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        return self.collect_data_from_loader(test_dataloader)

    def get_target_filenames(self) -> list[str]:
        """Return target (ground truth) data from the target folder (or testing dataloader)."""
        return os.listdir(os.path.join(self.results_dir, "targets"))

    def get_prediction_filenames(self) -> list[str]:
        """Return predictions data from the predictions folder (or model predict)."""
        return os.listdir(os.path.join(self.results_dir, "predictions"))

    def read_target(self, target_filename: str) -> torch.Tensor:
        """Read a target subject from specified path and return as torch tensor."""
        tgt_file_path = os.path.join(self.results_dir, "targets", target_filename)
        return torch.from_numpy(np.loadtxt(tgt_file_path))

    def read_prediction(self, pred_filename: str) -> torch.Tensor:
        """Read a predictions subject from specified path and return as torch tensor."""
        pred_file_path = os.path.join(self.results_dir, "predictions", pred_filename)
        return torch.from_numpy(np.loadtxt(pred_file_path))

    def get_testing_results_for_metric(
        self, metric: str = "mse", save: bool = False
    ) -> pd.DataFrame:
        """Get subject-wise results for the given metric."""
        targets = self.get_target_filenames()
        predictions = self.get_prediction_filenames()
        all_metrics = []
        for subject_pred, subject_tgt in zip(predictions, targets):
            pred = self.read_prediction(subject_pred)
            tgt = self.read_target(subject_tgt)
            if metric == "mse":
                result = torch.square(pred - tgt).mean()
            elif metric == "mae":
                result = torch.abs(pred - tgt).mean()
            elif metric == "frob":
                result = torch.sqrt(torch.square(pred - tgt).sum()).mean()
            elif metric == "pcc":
                eps = 1e-6
                pred_mean = pred - pred.mean(dim=1) + eps
                target_mean = tgt - tgt.mean(dim=1) + eps
                r_num = torch.sum(pred_mean * target_mean, dim=0)
                r_den = torch.sqrt(
                    torch.sum(torch.pow(pred_mean, 2), dim=0)
                    * torch.sum(torch.pow(target_mean, 2), dim=0)
                )
                pear = r_num / r_den
                result = torch.pow(pear - 1.0, 2).mean()
            else:
                ValueError("Metric not implemented!")
            all_metrics.append(result)

        metric_results = pd.DataFrame(torch.stack(all_metrics).numpy())
        print(f"Avg. {metric}:", torch.stack(all_metrics).mean().item())
        if save:
            metric_results.to_csv(
                self.results_dir + f"/{self.dataset}_{self.hemisphere}_{metric}.csv",
                index=False,
                header=False,
            )
        return metric_results

    def get_testing_results_all(self, save: bool = False) -> None:
        """Get results for all metrics."""
        self.get_testing_results_for_metric("mse", save)
        self.get_testing_results_for_metric("frob", save)
        self.get_testing_results_for_metric("pcc", save)

    def model_predict(self) -> None:
        """Abstract class to get outputs of the models."""

    def load_trained_model(self) -> None:
        """Abstract class to load models."""

    def read_test_outputs(self) -> None:
        """Abstract class to read outputs."""


class BenchmarkGrenolNet(BaseBenchmark):
    """Benchmark class to furher evaluate our model outputs."""

    def __init__(
        self, model_name: str, dataset: str, hemisphere: str, cortical_feature: int = 0
    ) -> None:
        super().__init__(model_name, dataset, hemisphere, cortical_feature)


class BenchmarkMGCNGAN(BaseBenchmark):
    """Benchmark class to evaluate MGCN-GAN outputs."""

    def __init__(
        self, model_name: str, dataset: str, hemisphere: str, cortical_feature: int = 0
    ) -> None:
        super().__init__(model_name, dataset, hemisphere, cortical_feature)

    def get_target_filenames(self) -> list[str]:
        """Return target (ground truth) data from the target folder (or testing dataloader)."""
        file_list = os.listdir(
            os.path.join(
                self.results_dir,
                "results",
                f"corticalfeature_{str(self.cortical_feature)}",
            )
        )
        return [f for f in file_list if "real" in f]

    def get_prediction_filenames(self) -> list[str]:
        """Return predictions data from the predictions folder (or model predict)."""
        file_list = os.listdir(
            os.path.join(
                self.results_dir,
                "results",
                f"corticalfeature_{str(self.cortical_feature)}",
            )
        )
        return [f for f in file_list if "gen" in f]

    def read_target(self, target_filename: str) -> torch.Tensor:
        """Read a target subject from specified path and return as torch tensor."""
        tgt_file_path = os.path.join(
            self.results_dir,
            "results",
            f"corticalfeature_{str(self.cortical_feature)}",
            target_filename,
        )
        return torch.from_numpy(np.loadtxt(tgt_file_path))

    def read_prediction(self, pred_filename: str) -> torch.Tensor:
        """Read a predictions subject from specified path and return as torch tensor."""
        pred_file_path = os.path.join(
            self.results_dir,
            "results",
            f"corticalfeature_{str(self.cortical_feature)}",
            pred_filename,
        )
        return torch.from_numpy(np.loadtxt(pred_file_path))


class BenchmarkHADA(BaseBenchmark):
    """Benchmark class to evaluate HADA outputs."""

    def __init__(
        self, model_name: str, dataset: str, hemisphere: str, cortical_feature: int = 0
    ) -> None:
        super().__init__(model_name, dataset, hemisphere, cortical_feature)

    def get_target_filenames(self) -> list[str]:
        """Return target (ground truth) data from the target folder (or testing dataloader)."""
        real_file_path = os.path.join(
            self.results_dir, f"corticalfeature_{self.cortical_feature}.csv"
        )
        return np.genfromtxt(real_file_path, delimiter=",").tolist()[:42]

    def get_prediction_filenames(self) -> list[str]:
        """Return predictions data from the predictions folder (or model predict)."""
        pred_file_path = os.path.join(
            self.results_dir, f"allpredTV_{self.cortical_feature}.csv"
        )
        return np.genfromtxt(pred_file_path, delimiter=",").tolist()[:42]

    def read_target(self, tgt_subj: str) -> torch.Tensor:
        """Read a target subject from specified path and return as torch tensor."""
        return torch.from_numpy(GraphBuilder.anti_vectorize(tgt_subj, 34))

    def read_prediction(self, pred_subj: str) -> torch.Tensor:
        """Read a predictions subject from specified path and return as torch tensor."""
        return torch.from_numpy(GraphBuilder.anti_vectorize(pred_subj, 34))


if __name__ == "__main__":
    # benchmarker = BenchmarkMGCNGAN("mgcn-gan", "openneuro", "left", cortical_feature=2)
    # print(benchmarker)
    # benchmarker.get_testing_results_all(save=True)
    benchmarker = BenchmarkGrenolNet("dgn-net", "candishare", "left")
    print(benchmarker)
    benchmarker.get_testing_results_all(save=True)
    # benchmarker = BenchmarkHADA("hada", "openneuro", "right", cortical_feature=2)
    # print(benchmarker)
    # benchmarker.get_testing_results_all(save=True)
    benchmarker = BenchmarkGrenolNet("grenol-net", "candishare", "left")
    print(benchmarker)
    benchmarker.get_testing_results_all(save=True)
