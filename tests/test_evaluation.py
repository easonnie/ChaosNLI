import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'distnli/src'))
from eval_scripts.get_ensemble_prediction import ensemble_predictions
from eval_scripts.prediction_io import infer_task, read_rows, standardized_row
from eval_scripts.temperature_calibration import calibrate, fit_temperature


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = [load_module('root_tools', 'src/evaluation/tools.py'),
         load_module('dist_tools', 'distnli/src/evaluation/tools.py')]
DOWNLOADER = load_module('download_data', 'scripts/download_data.py')


def example(uid, dist=None):
    return {'uid': uid, 'label_dist': dist or [0.6, 0.3, 0.1],
            'old_label': 'n', 'majority_label': 'e', 'entropy': 0.5}


def prediction(uid, label='entailment', probabilities=None):
    return {'uid': uid, 'predicted_label': label,
            'predicted_probabilities': probabilities or [0.6, 0.3, 0.1]}


class DistributionTests(unittest.TestCase):
    def test_softmax_is_stable_and_shift_invariant(self):
        for tools in TOOLS:
            with self.subTest(module=tools.__name__):
                np.testing.assert_allclose(tools.model_label_dist([1000, 1001, 1002]),
                                           tools.model_label_dist([-1000, -999, -998]))
                np.testing.assert_allclose(tools.model_label_dist([1e308, -1e308, 0]), [1, 1e-15, 1e-15])

    def test_normalization_preserves_input_and_integer_zeros(self):
        for tools in TOOLS:
            original = np.array([1, 0, 0])
            result = tools.normalize_dist(original)
            np.testing.assert_array_equal(original, [1, 0, 0])
            self.assertGreater(result[1], 0)
            self.assertAlmostEqual(result.sum(), 1)
            self.assertTrue(np.all(np.isfinite(tools.normalize_dist([1e308, 1e308, 0]))))

    def test_invalid_distributions_fail(self):
        for tools in TOOLS:
            for values in ([0, 0, 0], [-0.1, 0.5, 0.6], [float('nan'), 1, 0],
                           [float('inf'), 1, 0], [[0.5, 0.5]], [1]):
                with self.subTest(module=tools.__name__, values=values):
                    with self.assertRaises(ValueError):
                        tools.normalize_dist(values)
            with self.assertRaises(ValueError):
                tools.model_label_dist([float('nan'), 0, 0])

    def test_all_correct_is_intersection_and_order_independent(self):
        data = {uid: example(uid) for uid in ('x', 'y')}
        models = {
            'first': {'x': prediction('x', 'neutral'), 'y': prediction('y')},
            'second': {'x': prediction('x'), 'y': prediction('y')},
        }
        for tools in TOOLS:
            for order in (models, dict(reversed(list(models.items())))):
                results, correct = tools.calculate_divergence_bwt_model_human_simplify(data, order)
                self.assertEqual(correct, {'y'})
                self.assertEqual(results['first']['m_acc'], 0.5)
                self.assertEqual(results['second']['m_acc'], 1)

    def test_metrics_match_reference(self):
        human, model = [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]
        for tools in TOOLS:
            results, _ = tools.calculate_divergence_bwt_model_human_simplify(
                {'x': example('x', human)}, {'model': {'x': prediction('x', probabilities=model)}})
            self.assertAlmostEqual(results['model']['average JS div'], jensenshannon(human, model), places=13)
            self.assertAlmostEqual(results['model']['average KL div'], entropy(human, model), places=13)
            self.assertAlmostEqual(results['model']['o_acc'], 0)

    def test_identical_distributions_have_zero_distance(self):
        for tools in TOOLS:
            for dist in ([0.6, 0.3, 0.1], [1, 0, 0]):
                results, _ = tools.calculate_divergence_bwt_model_human_simplify(
                    {'x': example('x', dist)}, {'model': {'x': prediction('x', probabilities=dist)}})
                self.assertLess(abs(results['model']['average JS div']), 1e-7)

    def test_missing_predictions_and_empty_inputs_fail_clearly(self):
        for tools in TOOLS:
            with self.assertRaisesRegex(ValueError, 'missing 1 prediction.*x'):
                tools.calculate_divergence_bwt_model_human_simplify({'x': example('x')}, {'model': {}})
            for data, models in (({}, {'model': {}}), ({'x': example('x')}, {})):
                with self.assertRaises(ValueError):
                    tools.calculate_divergence_bwt_model_human_simplify(data, models)

    def test_invalid_dimensions_and_labels_are_rejected(self):
        for tools in TOOLS:
            for bad in (prediction('x', probabilities=[0.5, 0.5]), prediction('x', 'unknown')):
                with self.assertRaisesRegex(ValueError, 'UID.*x'):
                    tools.calculate_divergence_bwt_model_human_simplify(
                        {'x': example('x')}, {'model': {'x': bad}})

    def test_empty_all_correct_subset_has_undefined_metrics(self):
        for tools in TOOLS:
            results, correct = tools.calculate_divergence_bwt_model_human_simplify(
                {'x': example('x')}, {'model': {'x': prediction('x', 'neutral')}})
            self.assertFalse(correct)
            self.assertTrue(math.isnan(results['model']['correct example JS div']))

    def test_repeated_entropy_quantiles_and_custom_model_names(self):
        data = {uid: example(uid) for uid in ('x', 'y')}
        models = {'custom': {uid: prediction(uid) for uid in data}}
        for tools in TOOLS:
            partitions = tools.build_entropy_bins(data, 5, type='quantile')
            self.assertEqual(sum(len(items) for _, items in partitions), 2)
            results = tools.calculate_per_bin_results_simplify(partitions, models)
            frame = tools.build_bin_items(results)
            self.assertEqual(frame['Total Count'].sum(), 2)
            self.assertEqual(set(frame['Model']), {'custom'})

    def test_empty_entropy_bins_have_nan_metrics(self):
        for tools in TOOLS:
            data = {'x': example('x')}
            bins = tools.build_entropy_bins(data, 2, customized_bins=[0, 0.25, 1])
            result = tools.calculate_per_bin_results_simplify(bins, {'custom': {'x': prediction('x')}})
            self.assertTrue(math.isnan(result['custom']['bin_results'][(0, 0.25)]['m_acc']))

    def test_alphanli_label_order(self):
        data = {'x': {'uid': 'x', 'label_dist': [0.2, 0.8], 'old_label': 1, 'majority_label': 2}}
        models = {'model': {'x': {'uid': 'x', 'logits': [0, 1], 'predicted_label': 2}}}
        for tools in TOOLS:
            result, correct = tools.calculate_divergence_bwt_model_human_simplify(data, models, 'uncertainty_abdnli')
            self.assertEqual(correct, {'x'})
            self.assertEqual(result['model']['m_acc'], 1)


class PostprocessingTests(unittest.TestCase):
    def test_conversion_infers_task_from_dimensions_and_recomputes_labels(self):
        row = {'uid': 'x', 'logits': [0, 1, 0], 'predicted_label': 0}
        task = infer_task([row])
        self.assertEqual(standardized_row(row, task)['predicted_label'], 'neutral')
        alpha = {'uid': 'x', 'predicted_probabilities': [0.1, 0.9], 'predicted_label': 0}
        self.assertEqual(standardized_row(alpha, infer_task([alpha]))['predicted_label'], 2)

    def test_ensemble_label_matches_averaged_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for index, logits in enumerate(([0, 1, 0], [10, 0, 0])):
                (folder / '{}.jsonl'.format(index)).write_text(json.dumps(
                    {'uid': 'x', 'logits': logits, 'predicted_label': int(np.argmax(logits))}) + '\n')
            result = ensemble_predictions(folder)['x']
            self.assertEqual(result['predicted_label'], 'entailment')
            self.assertEqual(int(np.argmax(result['predicted_probabilities'])), 0)

    def test_ensemble_requires_identical_uid_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for index, uid in enumerate(('x', 'y')):
                (folder / '{}.jsonl'.format(index)).write_text(json.dumps(prediction(uid)) + '\n')
            with self.assertRaisesRegex(ValueError, 'identical UID'):
                ensemble_predictions(folder)

    def test_empty_ensemble_and_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(ValueError, 'No prediction'):
                ensemble_predictions(folder)
            path = folder / 'predictions.jsonl'
            path.write_text((json.dumps(prediction('x')) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_rows(path)

    def test_calibration_recovers_known_temperature(self):
        labels = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        logits = np.log(labels) * 2
        temperature = fit_temperature(logits, labels)
        self.assertAlmostEqual(temperature, 2, places=4)
        self.assertGreater(temperature, 0)

    def test_calibration_applies_existing_temperature_and_removes_stale_probabilities(self):
        models = {'custom': {'x': {'uid': 'x', 'predicted_probabilities': [0.8, 0.1, 0.1],
                                   'logits': [0, 10, 0], 'predicted_label': 'neutral'}}}
        temperature = calibrate(models, 'unused.jsonl', temperature=2)
        self.assertEqual(temperature, 2)
        result = models['custom']['x']
        self.assertNotIn('predicted_probabilities', result)
        self.assertEqual(result['predicted_label'], 'entailment')
        np.testing.assert_allclose(result['logits'], np.log([0.8, 0.1, 0.1]) / 2)

    def test_invalid_temperature_is_rejected(self):
        for temperature in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'positive'):
                calibrate({'m': {'x': prediction('x')}}, 'unused', temperature=temperature)

    def test_calibration_does_not_modify_other_models(self):
        other = {'x': prediction('x')}
        models = {'selected': copy.deepcopy(other), 'other': copy.deepcopy(other)}
        calibrate(models, 'unused', model_name='selected', temperature=2)
        self.assertEqual(models['other'], other)

    def test_calibration_rejects_invalid_human_distributions(self):
        with self.assertRaises(ValueError):
            fit_temperature([[1, 2, 3]], [[0.2, 0.3, 0.1]])


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.content = b'original payload\n'
        self.filename = 'sample.jsonl'
        self.output = Path(self.directory.name)
        self.checksums = mock.patch.dict(DOWNLOADER.CHECKSUMS,
                                        {self.filename: hashlib.sha256(self.content).hexdigest()}, clear=True)
        self.checksums.start()
        self.addCleanup(self.checksums.stop)

    def test_verified_cache_does_not_need_network(self):
        (self.output / self.filename).write_bytes(self.content)
        with mock.patch.object(DOWNLOADER, 'urlopen') as fetch, contextlib.redirect_stdout(io.StringIO()):
            DOWNLOADER.download_data(self.output)
            fetch.assert_not_called()

    def test_corrupt_cached_file_is_replaced_only_after_verification(self):
        path = self.output / self.filename
        path.write_bytes(b'corrupt')
        with mock.patch.object(DOWNLOADER, 'urlopen', return_value=io.BytesIO(self.content)), \
                contextlib.redirect_stdout(io.StringIO()):
            DOWNLOADER.download_data(self.output)
        self.assertEqual(path.read_bytes(), self.content)
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_checksum_failure_preserves_existing_file(self):
        path = self.output / self.filename
        path.write_bytes(b'existing file')
        with mock.patch.object(DOWNLOADER, 'urlopen', return_value=io.BytesIO(b'wrong payload')):
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                DOWNLOADER.download_data(self.output)
        self.assertEqual(path.read_bytes(), b'existing file')

    def test_network_failure_has_nonzero_status_and_no_success_message(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(DOWNLOADER, 'urlopen', side_effect=URLError('offline')), \
                mock.patch.object(sys, 'argv', ['download_data.py', '--output-dir', str(self.output)]), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                DOWNLOADER.main()
        self.assertEqual(error.exception.code, 1)
        self.assertNotIn('ChaosNLI ready', stdout.getvalue())
        self.assertIn('Download failed', stderr.getvalue())


class CliTests(unittest.TestCase):
    def run_cli(self, path, *args):
        return subprocess.run([sys.executable, str(ROOT / path), *map(str, args)],
                              cwd=tempfile.gettempdir(), capture_output=True, text=True)

    def test_both_evaluation_clis_work_without_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / 'examples.jsonl'
            preds = folder / 'predictions.json'
            data.write_text(json.dumps(example('x')) + '\n')
            preds.write_text(json.dumps({'custom': {'x': prediction('x')}}))
            for path in ('src/scripts/evaluate.py', 'distnli/src/eval_scripts/evaluate.py'):
                result = self.run_cli(path, '--task_name', 'uncertainty_nli', '--data_file', data,
                                      '--prediction_file', preds)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('custom', result.stdout)
                self.assertIn('All Correct Count: 1', result.stdout)

    def test_duplicate_dataset_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data, preds = folder / 'examples.jsonl', folder / 'predictions.json'
            data.write_text((json.dumps(example('x')) + '\n') * 2)
            preds.write_text(json.dumps({'custom': {'x': prediction('x')}}))
            result = self.run_cli('src/scripts/evaluate.py', '--task_name', 'uncertainty_nli',
                                  '--data_file', data, '--prediction_file', preds)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Duplicate UIDs', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_conversion_and_calibration_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw, unified, calibrated = [folder / name for name in ('raw.jsonl', 'unified.json', 'calibrated.json')]
            raw.write_text(json.dumps({'uid': 'x', 'logits': [1000, 1001, 1002], 'predicted_label': 0}) + '\n')
            result = self.run_cli('distnli/src/eval_scripts/unify_prediction_format.py', raw, unified,
                                  '--model_name', 'custom')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(unified.read_text())['custom']['x']['predicted_label'], 'contradiction')
            result = self.run_cli('distnli/src/eval_scripts/temperature_calibration.py', unified,
                                  folder / 'unused.jsonl', calibrated, 2)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(float(result.stdout), 2)
            np.testing.assert_allclose(json.loads(calibrated.read_text())['custom']['x']['logits'], [500, 500.5, 501])

    def test_unli_dispatch_uses_task_name_instead_of_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data, preds = folder / 'examples.jsonl', folder / 'predictions.json'
            rows = [{'uid': str(i), 'unli_label': p} for i, p in enumerate((0.1, 0.5, 0.8))]
            data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            predictions = {row['uid']: prediction(row['uid'], probabilities=[row['unli_label'],
                            (1-row['unli_label'])/2, (1-row['unli_label'])/2]) for row in rows}
            preds.write_text(json.dumps({'custom': predictions}))
            result = self.run_cli('distnli/src/eval_scripts/evaluate.py', '--task_name', 'unli',
                                  '--data_file', data, '--prediction_file', preds)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Pearson Corr', result.stdout)
            self.assertIn('1.0', result.stdout)


if __name__ == '__main__':
    unittest.main()
