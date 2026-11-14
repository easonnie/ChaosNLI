"""Shared file and label handling for prediction postprocessing."""

import json
import numpy as np
from evaluation.tools import _prediction_distribution


TASK_SIZES = {'uncertainty_nli': 3, 'uncertainty_abdnli': 2}
NLI_LABELS = ('entailment', 'neutral', 'contradiction')


def read_rows(path):
    rows = []
    with open(path, encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError('{}:{}: {}'.format(path, line_number, error)) from error
            if not isinstance(row, dict) or 'uid' not in row:
                raise ValueError('{}:{}: expected a prediction with a uid'.format(path, line_number))
            rows.append(row)
    if not rows:
        raise ValueError('Prediction file is empty: ' + str(path))
    ids = [row['uid'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prediction UIDs in ' + str(path))
    return rows


def infer_task(rows, task_name=None):
    sizes = {len(_prediction_distribution(row)) for row in rows}
    if len(sizes) != 1:
        raise ValueError('All predictions must have the same number of classes')
    size = sizes.pop()
    if task_name is None:
        for task, expected in TASK_SIZES.items():
            if size == expected:
                return task
        raise ValueError('Only two-choice alphaNLI and three-way NLI are supported')
    if TASK_SIZES[task_name] != size:
        raise ValueError('Prediction dimensions do not match task_name=' + task_name)
    return task_name


def predicted_label(probabilities, task_name):
    index = int(np.argmax(probabilities))
    return NLI_LABELS[index] if task_name == 'uncertainty_nli' else index + 1


def standardized_row(row, task_name):
    probabilities = _prediction_distribution(row, TASK_SIZES[task_name])
    result = dict(row)
    # Use the distribution as the source of truth for labels, including MC outputs.
    result['predicted_label'] = predicted_label(probabilities, task_name)
    return result


def read_prediction_dict(path):
    with open(path, encoding='utf-8') as source:
        predictions = json.load(source)
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError('Expected a nonempty JSON mapping of models to predictions')
    for name, rows in predictions.items():
        if not isinstance(rows, dict) or not rows:
            raise ValueError('Expected a nonempty UID mapping for model ' + str(name))
        for uid, row in rows.items():
            if not isinstance(row, dict) or row.get('uid', uid) != uid:
                raise ValueError('Invalid prediction for UID ' + str(uid))
    return predictions


def write_prediction_dict(path, predictions):
    # A single JSON object also remains compatible with the original one-line JSONL readers.
    with open(path, 'w', encoding='utf-8') as output:
        json.dump(predictions, output, allow_nan=False)
        output.write('\n')
