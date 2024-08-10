"""Module to define plotting utilities for graphs."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from avicortex.datasets import GraphDataset
from torch_geometric.data import Data as PygData


def plot_graph_adjacency(
    graph_list: list[PygData],
    titles: list[Any],
    batch: int = 0,
    view: int = 0,
    figsize: float | None = None,
) -> None:
    """Plot adjacency matrices of the given graphs in a list."""
    n_plots = len(graph_list)
    g_matrices = []
    for g in graph_list:
        g_adj = g.edge_attr[batch, :, view].reshape(34, 34).cpu().detach().numpy()
        g_matrices.append(g_adj)
    if figsize is not None:
        _fig, axes = plt.subplots(1, n_plots, figsize=(figsize, figsize * n_plots))
    else:
        _fig, axes = plt.subplots(1, n_plots)

    for ax, g, title in zip(axes, g_matrices, titles):
        ax.matshow(g)
        ax.title.set_text(title)
        ax.set_axis_off()
        # fig.colorbar(subplot, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_results_boxplot(
    out_path: str, hem: str = "left", metric: str = "frob", size_multiplier: int = 3
) -> None:
    """Plot the given benchmarks in a boxplot."""
    dataset_names = ["candishare", "hcp", "openneuro"]
    if out_path.endswith("cross-cohort"):
        benchmarks = ["dgn-net", "grenol-net"]
        custom_palette = {
            "dgn-net": "#477CA8",
            "grenol-net": "#905998",
        }
    else:
        benchmarks = ["mgcn-gan", "dgn-net", "hada", "grenol-net"]
        custom_palette = {
            "mgcn-gan": "#CB3335",
            "dgn-net": "#477CA8",
            "hada": "#59A257",
            "grenol-net": "#905998",
        }

    # Collect data
    all_dfs = []
    for benchmark in benchmarks:
        for dataset_name in dataset_names:
            path = os.path.join(
                out_path,
                benchmark,
                f"{dataset_name}_{hem}",
                f"{dataset_name}_{hem}_{metric}.csv",
            )
            df = pd.DataFrame()
            result_arr = pd.read_csv(path, header=None).values

            if dataset_name == "openneuro":
                # We only train models with baseline subjects.
                # Workaround: the first half should be the baseline subjects.
                result_arr = result_arr[:42]

            data_len = result_arr.shape[0]

            if benchmark in {"hada", "mgcn-gan"}:
                # Workaround since these benchmarks has different splits.
                # Selecting the same subjects as test set for these methods needed.
                _, test_ind = GraphDataset.get_fold_indices(data_len, 2, 0)
                result_arr = result_arr[test_ind]
            df["Metric"] = np.squeeze(result_arr)
            df["Dataset"] = dataset_name
            df["Benchmark"] = benchmark
            if benchmark == "grenol-net":
                df["Benchmark"] + df["Benchmark"] + " (ours)"
            all_dfs.append(df)
    all_combined = pd.concat(all_dfs).reset_index(drop=True)

    sns.set(
        context="paper",
        style="darkgrid",
        rc={
            "figure.dpi": 100 * size_multiplier,
            "figure.figsize": (10, 5),
            # "axes.titlesize": 20 / size_multiplier,
            # "axes.labelsize": 20 / size_multiplier,
            # "axes.linewidth": 1.0 / size_multiplier,
            "xtick.labelsize": 25.0,
            # "xtick.major.pad": 3.5 / size_multiplier,
            # "xtick.minor.pad": 3.4 / size_multiplier,
            "ytick.labelsize": 15.0,
            # "ytick.major.pad": 3.5 / size_multiplier,
            # "ytick.minor.pad": 3.4 / size_multiplier,
            # "legend.fontsize": 20 / size_multiplier,
            # "legend.title_fontsize": 20 / size_multiplier,
            # "figure.titlesize": 20 / size_multiplier,
            # "lines.linewidth": 1.0 / size_multiplier,
            # "lines.markersize": 6.0 / size_multiplier,
            # "boxplot.flierprops.markersize": 6.0 / size_multiplier,
            # "boxplot.flierprops.markeredgewidth": 1.0 / size_multiplier,
            # "boxplot.flierprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.boxprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.whiskerprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.capprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.medianprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.meanprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.meanprops.markersize": 6.0 / size_multiplier,
            # "boxplot.showmeans": True,
        },
    )

    # Create and display the plot
    ax = sns.boxplot(
        x="Dataset",
        y="Metric",
        hue="Benchmark",
        data=all_combined,
        # palette="Set1",
        palette=custom_palette,
        width=0.8,
        dodge=True,
        legend=None,
    )
    # ax = sns.swarmplot(
    #     x="Dataset", y="Metric", hue="Benchmark", data=all_combined, dodge=True
    # )
    if metric == "frob":
        ax.set_ylim([0.0, 5.0])
    if metric == "mse":
        ax.set_ylim([0.0, 0.02])
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{hem}_{metric}.png")
    plt.close()
    all_combined.to_csv("./all_results_plotting.csv", index=False)


if __name__ == "__main__":
    plot_results_boxplot("./cross-cohort", size_multiplier=5, hem="left", metric="frob")
    plot_results_boxplot(
        "./cross-cohort", size_multiplier=5, hem="right", metric="frob"
    )
