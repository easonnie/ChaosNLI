"""Convert raw prediction JSONL to the ChaosNLI evaluation format."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_scripts.prediction_io import TASK_SIZES, infer_task, read_rows, standardized_row, write_prediction_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction_file', type=Path)
    parser.add_argument('output_file', type=Path)
    parser.add_argument('--task_name', choices=tuple(TASK_SIZES))
    parser.add_argument('--model_name', default='roberta-large')
    args = parser.parse_args()
    try:
        rows = read_rows(args.prediction_file)
        task_name = infer_task(rows, args.task_name)
        predictions = {row['uid']: standardized_row(row, task_name) for row in rows}
        write_prediction_dict(args.output_file, {args.model_name: predictions})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Conversion failed: {}\n'.format(error))


if __name__ == '__main__':
    main()
