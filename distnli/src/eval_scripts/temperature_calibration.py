"""Fit a positive calibration temperature on CPU, then scale prediction logits."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.tools import _prediction_distribution, _vector
from eval_scripts.prediction_io import (
    infer_task, predicted_label, read_prediction_dict, read_rows, write_prediction_dict,
)


def prediction_logits(prediction):
    if 'predicted_probabilities' in prediction:
        # The evaluator also prefers probabilities when both fields are present.
        return np.log(_prediction_distribution(prediction))
    return _vector(prediction['logits'], 'logits')


def fit_temperature(logits, labels):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if (logits.ndim != 2 or logits.shape != labels.shape or not len(logits)
            or not np.all(np.isfinite(logits)) or not np.all(np.isfinite(labels))
            or np.any(labels < 0) or not np.allclose(labels.sum(axis=1), 1)):
        raise ValueError('Calibration requires finite, aligned logits and normalized label distributions')
    # Optimizing log(T) guarantees T > 0. Cross-entropy and KL have the same optimum.
    def objective(log_temperature):
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = logits / np.exp(log_temperature)
            value = -np.sum(labels * log_softmax(scaled, axis=1)) / len(labels)
        return float(value) if np.isfinite(value) else float('inf')

    result = minimize_scalar(objective, bounds=(-np.log(1000), np.log(1000)), method='bounded')
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('Temperature optimization failed')
    # Keep the uncalibrated model if it already minimizes the objective.
    return float(np.exp(result.x)) if result.fun < objective(0.0) else 1.0


def calibrate(predictions, label_file, model_name=None, temperature=None):
    if model_name is None:
        if len(predictions) != 1:
            raise ValueError('Use --model_name when the input contains multiple models')
        model_name = next(iter(predictions))
    if model_name not in predictions:
        raise ValueError('Unknown model: ' + model_name)
    rows = predictions[model_name]
    task_name = infer_task(list(rows.values()))
    if temperature is None:
        labels = read_rows(label_file)
        missing = [label['uid'] for label in labels if label['uid'] not in rows]
        if missing:
            raise ValueError('Calibration predictions missing for UID ' + str(missing[0]))
        logits = [prediction_logits(rows[label['uid']]) for label in labels]
        temperature = fit_temperature(logits, [label['label_dist'] for label in labels])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    for uid, row in rows.items():
        with np.errstate(over='ignore', invalid='ignore'):
            logits = prediction_logits(row) / temperature
        if not np.all(np.isfinite(logits)):
            raise ValueError('Scaled logits are not finite for UID ' + str(uid))
        row['logits'] = logits.tolist()
        row.pop('predicted_probabilities', None)
        row['predicted_label'] = predicted_label(logits, task_name)
    return temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction_file', type=Path)
    parser.add_argument('label_file', type=Path)
    parser.add_argument('output_file', type=Path)
    parser.add_argument('temperature', nargs='?', type=float)
    parser.add_argument('--model_name')
    args = parser.parse_args()
    try:
        predictions = read_prediction_dict(args.prediction_file)
        temperature = calibrate(predictions, args.label_file, args.model_name, args.temperature)
        write_prediction_dict(args.output_file, predictions)
        print(temperature)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Calibration failed: {}\n'.format(error))


if __name__ == '__main__':
    main()
