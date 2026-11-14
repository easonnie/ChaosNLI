"""Average model probabilities and derive labels from the ensemble distribution."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.tools import _prediction_distribution
from eval_scripts.prediction_io import TASK_SIZES, infer_task, predicted_label, read_rows, write_prediction_dict


def ensemble_predictions(input_dir, task_name=None):
    paths = sorted(Path(input_dir).glob('*.jsonl'))
    if not paths:
        raise ValueError('No prediction JSONL files found in ' + str(input_dir))
    accumulated = {}
    expected_uids = None
    for path in paths:
        rows = read_rows(path)
        current_task = infer_task(rows, task_name)
        if task_name is None:
            task_name = current_task
        uids = {row['uid'] for row in rows}
        if expected_uids is None:
            expected_uids = uids
        elif uids != expected_uids:
            raise ValueError('Ensemble members must contain identical UID sets: ' + str(path))
        for row in rows:
            uid = row['uid']
            probabilities = _prediction_distribution(row, TASK_SIZES[task_name])
            accumulated[uid] = accumulated.get(uid, np.zeros_like(probabilities)) + probabilities
    results = {}
    for uid, total in accumulated.items():
        probabilities = total / len(paths)
        results[uid] = {
            'uid': uid, 'predicted_probabilities': probabilities.tolist(),
            'predicted_label': predicted_label(probabilities, task_name),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_file', type=Path)
    parser.add_argument('--task_name', choices=tuple(TASK_SIZES))
    parser.add_argument('--model_name', default='roberta-large')
    args = parser.parse_args()
    try:
        if args.output_file.resolve().parent == args.input_dir.resolve() and args.output_file.suffix == '.jsonl':
            raise ValueError('Write the ensemble output outside the member JSONL directory')
        predictions = ensemble_predictions(args.input_dir, args.task_name)
        write_prediction_dict(args.output_file, {args.model_name: predictions})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Ensemble failed: {}\n'.format(error))


if __name__ == '__main__':
    main()
