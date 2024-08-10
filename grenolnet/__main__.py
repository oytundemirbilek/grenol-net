"""Entrypoint of the training and inference scripts."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from grenolnet import __version__
from grenolnet.benchmarking import (
    BaseBenchmark,
    BenchmarkGrenolNet,
    BenchmarkHADA,
    BenchmarkMGCNGAN,
)
from grenolnet.experiments import Experiment
from grenolnet.plotting import plot_results_boxplot
from grenolnet.stattests import BaseStatTest


def parse_args() -> Namespace:
    """Parse command line arguments and return as dictionary."""
    parser = ArgumentParser(
        prog="Grenol-Net",
        description="Predict a target brain graph using a source brain graph.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    main_args = parser.add_argument_group("main options")
    infer_args = parser.add_argument_group("test options")
    data_args = parser.add_argument_group("dataset options")
    eval_args = parser.add_argument_group("evaluation options")
    main_args.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="run the training loop on the target dataset.",
    )
    main_args.add_argument(
        "-i",
        "--infer",
        action="store_true",
        help="run inference loop on the target dataset for either testing or inference purposes.",
    )
    main_args.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="run benchmarking method on the target dataset. Results should be available from --infer.",
    )
    main_args.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="plot results from benchmarking. Results should be available from --benchmark.",
    )
    main_args.add_argument(
        "-s",
        "--stat-test",
        action="store_true",
        help="run statistical test method on the target dataset. Results should be available from --infer.",
    )
    main_args.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="nameless_model",
        help="model name to be loaded and used for testing/inference.",
    )
    main_args.add_argument(
        "-de",
        "--device",
        type=str,
        default=None,
        help="The device that model will use.",
        choices=["cpu", "cuda"],
    )
    # ---------------------------------------------------------------------------------------------
    # Testing/inference args
    # ---------------------------------------------------------------------------------------------
    infer_args.add_argument(
        "-l",
        "--load-model",
        action="store_true",
        help="whether to load from the pretrained model files, in --infer mode.",
    )
    infer_args.add_argument(
        "-is",
        "--save-each",
        action="store_true",
        help="whether to store every predicted graph for each subject, in --infer mode.",
    )
    infer_args.add_argument(
        "-f",
        "--fold-id",
        type=int,
        default=None,
        help="from which dataset fold the model is trained on, in --infer mode.",
    )
    infer_args.add_argument(
        "-r",
        "--get-table",
        action="store_true",
        help="whether to store results in a table at the end, in --infer mode.",
    )
    # ---------------------------------------------------
    # Dataset args
    # ---------------------------------------------------
    data_args.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="candishare",
        help="dataset name which the operations will be based on.",
        choices=["candishare", "openneuro", "hcp"],
    )
    data_args.add_argument(
        "-dh",
        "--hem",
        type=str,
        default="left",
        help="brain hemisphere of which the dataset will be built from.",
        choices=["left", "right"],
    )
    # ---------------------------------------------------
    # Evaluation args
    # ---------------------------------------------------
    eval_args.add_argument(
        "-eb",
        "--target-benchmark",
        type=str,
        default="grenol-net",
        help="which benchmark model to compare our model.",
        choices=["grenol-net", "hada", "dgn-net", "mgcn-gan"],
    )
    eval_args.add_argument(
        "-em",
        "--eval-metric",
        type=str,
        default="mse",
        help="which evaluation metric should be used when plotting or statistical testing.",
        choices=["mse", "frob"],
    )
    eval_args.add_argument(
        "-ep",
        "--benchmark-path",
        type=str,
        default="./",
        help="benchmark path which will be used to create plots from benchmarking results.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute experiments entrypoint."""
    args = parse_args()

    exp = Experiment(
        hemisphere=args.hem,
        dataset=args.dataset,
        timepoint="baseline",
        n_epochs=500,
        n_folds=5,
        batch_size=100,
        validation_period=5,
        learning_rate=0.001,
        loss_name="mse",
        diffusion_step_count=100,
        diffusion_noise_schedule="cosine",
        diffusion_noise_dist="normal",
        diffusion_noise_mean=0.0,
        diffusion_noise_std=0.01,
        diffusion_noise_max=0.02,
        src_feature_idx=0,
        tgt_feature_idx=2,
        loss_weight=0.5,
        conv_size=48,
        model_name=args.model_name,
        random_seed=0,
        device=args.device,
    )

    if args.train:
        exp.train_model()

    if args.infer:
        exp.run_inference(
            mode="test",
            load=args.load_model,
            save_predictions=args.save_each,
            fold=args.fold_id,
        )

    if args.get_table:
        exp.get_results_table()

    # Post methods after training-testing is finished.
    if args.benchmark:
        benchmarker: BaseBenchmark
        if args.target_benchmark == "hada":
            benchmarker = BenchmarkHADA("hada", args.dataset, args.hem, 2)
            print(benchmarker)
            benchmarker.get_testing_results_all(save=True)
        elif args.target_benchmark == "mgcn-gan":
            benchmarker = BenchmarkMGCNGAN("mgcn-gan", args.dataset, args.hem, 2)
            print(benchmarker)
            benchmarker.get_testing_results_all(save=True)
        else:
            benchmarker = BenchmarkGrenolNet(
                args.target_benchmark, args.dataset, args.hem
            )
            print(benchmarker)
            benchmarker.get_testing_results_all(save=True)

    if args.plot:
        plot_results_boxplot(args.benchmark_path, args.hem, args.eval_metric)

    if args.stat_test:
        tester = BaseStatTest(
            dataset=args.dataset,
            hemisphere=args.hem,
            metric=args.eval_metric,
        )
        print(tester)
        tester.run()


if __name__ == "__main__":
    main()
