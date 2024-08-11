"""Module to generalize statistical hypothesis testing."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from avicortex.datasets import GraphDataset

FILE_DIR = os.path.dirname(__file__)


class BaseStatTest:
    """Class for standard statistical tests."""

    def __init__(self, dataset: str, hemisphere: str, metric: str = "frob") -> None:
        self.metric = metric
        self.dataset = dataset
        self.hem = hemisphere
        dataset_names = ["candishare", "hcp", "openneuro"]
        if dataset not in set(dataset_names):
            raise ValueError("Available datasets are 'candishare', 'hcp', 'openneuro'.")
        if hemisphere not in {"left", "right"}:
            raise ValueError("Available hemisphere are 'left' and 'right'.")

        self.root_dir = os.path.join(FILE_DIR, "..", "benchmarks")
        self.results_dir = os.path.join(self.root_dir, dataset + "_" + hemisphere)

    def __repr__(self) -> str:
        """Return string representation of the class."""
        return (
            f"{self.__class__.__name__} statistical test - {self.dataset} - {self.hem}"
        )

    def run(self) -> None:
        """Run the statistical tests, save the results."""
        # benchmarks = ["dgn-net", "grenol-net"]
        benchmarks = ["mgcn-gan", "dgn-net", "hada", "grenol-net"]

        # Collect data
        all_results_dataset = []
        for benchmark in benchmarks:
            path = os.path.join(
                self.root_dir,
                benchmark,
                f"{self.dataset}_{self.hem}",
                f"{self.dataset}_{self.hem}_{self.metric}.csv",
            )
            result_arr = pd.read_csv(path, header=None).values
            # We only train models with baseline subjects.
            # Workaround: the first half should be the baseline test subjects.
            data_len = (
                result_arr.shape[0] // 2
                if self.dataset == "openneuro"
                else result_arr.shape[0]
            )
            if benchmark in {"hada", "mgcn-gan"}:
                # Workaround since these benchmarks has different splits.
                # Selecting the same subjects as test set for these methods needed.
                _, test_ind = GraphDataset.get_fold_indices(data_len, 2, 0)
                result_arr = result_arr[test_ind]
            all_results_dataset.append(np.squeeze(result_arr))

        all_results_arr = np.array(all_results_dataset)
        pval_matrix = sp.posthoc_conover_friedman(all_results_arr.T)
        print(pval_matrix)


if __name__ == "__main__":
    tester = BaseStatTest(dataset="hcp", hemisphere="right")
    print(tester)
    tester.run()
