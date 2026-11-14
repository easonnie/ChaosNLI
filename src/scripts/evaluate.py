import argparse
from pathlib import Path
import sys

# Prefer this project's modules even if another setup.sh was sourced earlier.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.model_pref import model_perf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        choices=("uncertainty_nli", "uncertainty_abdnli"),
        help="Task Name: (uncertainty_nli / uncertainty_abdnli).",
        required=True
    )

    parser.add_argument(
        "--data_file",
        type=str,
        help='The path to the data file.', required=True)

    parser.add_argument(
        "--prediction_file",
        type=str,
        default=None,
        help='The path to the prediction file.', required=True)

    args = parser.parse_args()

    dataset_name = ""
    if "chaosNLI_snli.jsonl" in args.data_file:
        dataset_name = "ChaosNLI-SNLI"
    elif "chaosNLI_mnli_m.jsonl" in args.data_file:
        dataset_name = "ChaosNLI-MNLI"
    elif "chaosNLI_alphanli.jsonl" in args.data_file:
        dataset_name = "ChaosNLI-alphaNLI"

    task_name = args.task_name
    data_file = args.data_file
    model_prediction_file = args.prediction_file
    try:
        model_perf(dataset_name, task_name, data_file, model_prediction_file)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Evaluation failed: {}\n".format(error))
