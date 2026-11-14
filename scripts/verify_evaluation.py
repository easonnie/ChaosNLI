"""Verify downloads, scoring, and CPU postprocessing on the real ChaosNLI data."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from download_data import CHECKSUMS, download_data


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('snli', 'snli', 'uncertainty_nli', 1514, (0.4541, 1.2947, 0.3283, 0.3289)),
    ('mnli_m', 'mnli', 'uncertainty_nli', 1599, (0.4010, 1.0460, 0.3802, 0.3827)),
    ('alphanli', 'abdnli', 'uncertainty_abdnli', 1532, (0.3711, 0.9964, 0.4693, 0.4713)),
)


def run_script(script, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise ValueError('{} failed:\n{}'.format(script, result.stderr))
    if 'RuntimeWarning' in result.stderr:
        raise ValueError('{} emitted a numerical warning:\n{}'.format(script, result.stderr))
    return result.stdout


def evaluate(data_file, prediction_file, task):
    outputs = [run_script(
        script, '--task_name', task, '--data_file', data_file,
        '--prediction_file', prediction_file,
    ) for script in ('src/scripts/evaluate.py', 'distnli/src/eval_scripts/evaluate.py')]
    if outputs[0] != outputs[1]:
        raise ValueError('Root and distnli evaluation results differ')
    return outputs[0]


def verify_postprocessing(work_dir, data, predictions, task):
    members = work_dir / 'members'
    members.mkdir(parents=True)
    for index, temperature in enumerate((1.0, 1.5)):
        rows = []
        for uid, prediction in predictions.items():
            logits = np.log(prediction['predicted_probabilities']) / temperature
            rows.append({'uid': uid, 'logits': logits.tolist(),
                         'predicted_label': int(np.argmax(logits))})
        (members / '{}.jsonl'.format(index)).write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8',
        )
    unified, ensemble, calibrated = [work_dir / (name + '.json')
                                     for name in ('unified', 'ensemble', 'calibrated')]
    calibration_labels = work_dir / 'calibration.jsonl'
    held_out = work_dir / 'held_out.jsonl'
    calibration_labels.write_text(''.join(json.dumps(row) + '\n' for row in data[:100]), encoding='utf-8')
    held_out.write_text(''.join(json.dumps(row) + '\n' for row in data[100:]), encoding='utf-8')
    run_script('distnli/src/eval_scripts/unify_prediction_format.py', members / '0.jsonl', unified)
    run_script('distnli/src/eval_scripts/get_ensemble_prediction.py', members, ensemble)
    temperature = float(run_script(
        'distnli/src/eval_scripts/temperature_calibration.py', ensemble, calibration_labels, calibrated,
    ))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Calibration did not produce a positive finite temperature')
    for prediction_file in (unified, ensemble, calibrated):
        evaluate(held_out, prediction_file, task)


def verify(data_dir, work_dir):
    for filename, checksum in CHECKSUMS.items():
        if hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() != checksum:
            raise ValueError('Checksum mismatch: ' + filename)
    print('All four dataset checksums match the pinned release.', flush=True)
    for suffix, sample, task, count, expected in CASES:
        data_file = data_dir / ('chaosNLI_' + suffix + '.jsonl')
        data = [json.loads(line) for line in data_file.read_text(encoding='utf-8').splitlines()]
        prediction_file = ROOT / 'data/prediction_samples' / (sample + '_random_baseline.json')
        predictions = json.loads(prediction_file.read_text(encoding='utf-8'))['random_baseline']
        if len(data) != count or {row['uid'] for row in data} != set(predictions):
            raise ValueError('Unexpected example count or prediction IDs: ' + suffix)
        output = evaluate(data_file, prediction_file, task)
        metrics_line = next(line for line in output.splitlines() if line.startswith('random_baseline'))
        actual = [float(value) for value in metrics_line.split()[1:]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-5)
        print('{}: {} examples; both evaluators match the reference scores.'.format(suffix, count), flush=True)
        if suffix in ('snli', 'alphanli'):
            verify_postprocessing(work_dir / suffix, data, predictions, task)
            print('{}: conversion, ensemble, CPU calibration, and held-out scoring passed.'.format(suffix), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Use existing data instead of downloading a fresh copy.')
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix='chaosnli-verify-') as directory:
            work_dir = Path(directory)
            data_dir = args.data_dir.resolve() if args.data_dir else work_dir / 'data'
            if args.data_dir is None:
                download_data(data_dir)
            verify(data_dir, work_dir)
    except (OSError, ValueError, AssertionError, subprocess.TimeoutExpired) as error:
        parser.exit(1, 'Verification failed: {}\n'.format(error))
    print('Evaluation verification passed.')


if __name__ == '__main__':
    main()
