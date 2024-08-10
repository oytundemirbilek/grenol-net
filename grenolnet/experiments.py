"""Module to wrap experiments, automatically handle cross validation and collect results."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from torch import Tensor
from torch.nn import Module

from grenolnet.inference import BaseInferer
from grenolnet.train import BaseTrainer

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.results_save_path = os.path.join(
            FILE_PATH, "..", "results", self.kwargs["model_name"]
        )
        self.model_per_fold: list[Module] = []
        self.last_val_result_per_fold: list[float] = []
        self.best_val_result_per_fold: list[float] = []
        self.test_result_per_fold: list[float] = []
        self.n_folds = self.kwargs.get("n_folds", 5)
        self.trainer = BaseTrainer(**self.kwargs)

    def train_model(
        self,
        fold: int | None = None,
    ) -> None:
        """Train cross validation models, and save the trained model."""
        if fold is None:
            iter_folds = self.n_folds
            start_fold = 0
        else:
            start_fold = fold
            iter_folds = fold + 1
        for fold_id in range(start_fold, iter_folds):
            print(f"--------------------- FOLD {fold_id} ---------------------")
            self.trainer = BaseTrainer(**self.kwargs)
            model = self.trainer.train(current_fold=fold_id)
            self.model_per_fold.append(model)
            # Last epochs validation score:
            self.last_val_result = self.trainer.val_loss_per_epoch[-1]
            self.best_val_result = self.trainer.best_val_loss
            self.last_val_result_per_fold.append(self.last_val_result)
            self.best_val_result_per_fold.append(self.best_val_result)

    def select_model(self) -> None:
        """Post-process to combine trained cross validation models to be used later for inference."""

    def run_inference(
        self,
        load: bool = False,
        mode: str = "test",
        save_predictions: bool = False,
        fold: int | None = None,
    ) -> None:
        """Run inference pipeline based on the defined training parameters."""
        self.all_results: dict[str, list[float]] = {}
        if fold is None:
            iter_folds = self.n_folds
            start_fold = 0
        else:
            start_fold = fold
            iter_folds = fold + 1
        for fold_id in range(start_fold, iter_folds):
            diff_params = {
                "step_count": self.trainer.diffusion_step_count,
                "noise_schedule": self.trainer.diffusion_noise_schedule,
                "noise_dist": self.trainer.diffusion_noise_dist,
                "noise_mean": self.trainer.diffusion_noise_mean,
                "noise_std": self.trainer.diffusion_noise_std,
                "noise_max": self.trainer.diffusion_noise_max,
            }
            if load:
                model_params = {
                    "conv_size": self.trainer.conv_size,
                    "in_views": 1,
                    "out_views": 1,
                    "n_nodes": 34,
                }
                self.inferer = BaseInferer(
                    dataset=self.kwargs["dataset"],
                    hemisphere=self.kwargs["hemisphere"],
                    src_feature_idx=self.kwargs["src_feature_idx"],
                    tgt_feature_idx=self.kwargs["tgt_feature_idx"],
                    model_path=self.trainer.model_save_path + f"_fold{fold_id}",
                    model_params=model_params,
                    diffusion_params=diff_params,
                    random_seed=self.kwargs["random_seed"],
                )
            else:
                # model_fold is a tuple of 2 models, first source, second target model
                self.inferer = BaseInferer(
                    dataset=self.kwargs["dataset"],
                    hemisphere=self.kwargs["hemisphere"],
                    src_feature_idx=self.kwargs["src_feature_idx"],
                    tgt_feature_idx=self.kwargs["tgt_feature_idx"],
                    model=self.model_per_fold[fold_id],
                    diffusion_params=diff_params,
                    random_seed=self.kwargs["random_seed"],
                )
            self.test_results = self.inferer.run(mode, save_predictions)
            self._results_tensor_to_list(self.test_results)
            print(self.all_results)

    def _results_tensor_to_list(
        self, scores: dict[str, Tensor]
    ) -> dict[str, list[float]]:
        for key in scores:  # noqa: PLC0206
            if "avg_" + key in self.all_results:
                self.all_results["avg_" + key].append(scores[key].mean().item())
            else:
                self.all_results["avg_" + key] = [scores[key].mean().item()]

            if "std_" + key in self.all_results:
                self.all_results["std_" + key].append(scores[key].std().item())
            else:
                self.all_results["std_" + key] = [scores[key].std().item()]
        return self.all_results

    def get_results_table(self) -> None:
        """Collect results in a table after inference pipeline run on testing set."""
        # TODO: Test score per fold or select a model then measure test score?
        results_df = pd.DataFrame(self.all_results)
        print(results_df)
        inference_dataset = self.kwargs["dataset"] + "_" + self.kwargs["hemisphere"]
        if len(self.last_val_result_per_fold) > 0:
            results_df["Last Val. Scores"] = self.last_val_result_per_fold
            results_df["Best Val. Scores"] = self.best_val_result_per_fold
        # Index of the dataframe will indicate the fold id.
        results_df.to_csv(
            self.results_save_path + f"_{inference_dataset}.csv", index=True
        )
