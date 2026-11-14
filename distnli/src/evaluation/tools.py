import numpy as np
from scipy.stats import entropy
import pandas as pd


def format_number(num, digits=4):
    return str(round(num, digits))


model_name_map_dict = {
    'bert-base': 'BERT-base', 'bert-large': 'BERT-large',
    'xlnet-base': 'XLNet-base', 'xlnet-large': 'XLNet-large',
    'roberta-base': 'RoBERTa-base', 'roberta-large': 'RoBERTa-large',
    'distilbert': 'DistilBert', 'albert-xxlarge': 'ALBERT', 'bart-large': 'BART',
}
simply_nli_label_mapping = {
    'e': 'e', 'c': 'c', 'n': 'n', 'entailment': 'e',
    'contradiction': 'c', 'neutral': 'n', '-': '-',
}
simply_nli_label_mapping_model_output_to_original = {
    'e': 'entailment', 'c': 'contradiction', 'n': 'neutral',
}
simply_abdnli_label_mapping_model_output_to_original = {1: 1, 2: 2}


def _vector(values, name):
    values = np.array(values, dtype=float, copy=True)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(name + ' must be a finite one-dimensional vector of at least two values')
    return values


def model_label_dist(logits_list):
    logits = _vector(logits_list, 'logits')
    # Subtract the largest logit before exponentiating to prevent overflow.
    with np.errstate(over='ignore', under='ignore'):
        probabilities = np.exp(logits - np.max(logits))
    return normalize_dist(probabilities)


def normalize_dist(prob):
    prob = _vector(prob, 'probabilities')
    if np.any(prob < 0) or np.max(prob) <= 0:
        raise ValueError('probabilities must be nonnegative and have a positive sum')
    # Scale first so even large finite weights cannot overflow their sum.
    prob /= np.max(prob)
    prob /= np.sum(prob)
    prob = np.maximum(prob, 1e-15)
    return prob / np.sum(prob)


def _prediction_distribution(prediction, size=None):
    if 'predicted_probabilities' in prediction:
        prob = normalize_dist(prediction['predicted_probabilities'])
    elif 'logits' in prediction:
        prob = model_label_dist(prediction['logits'])
    else:
        raise ValueError('Each prediction needs predicted_probabilities or logits')
    if size is not None and len(prob) != size:
        raise ValueError('Expected {} classes, got {}'.format(size, len(prob)))
    return prob


def _validate_predictions(items, model_prediction_dict):
    if not items:
        raise ValueError('The evaluation dataset is empty')
    if not isinstance(model_prediction_dict, dict) or not model_prediction_dict:
        raise ValueError('Predictions must be a nonempty mapping of model names to UID mappings')
    for model_name, predictions in model_prediction_dict.items():
        if not isinstance(predictions, dict):
            raise ValueError('Predictions for {!r} must be a UID mapping'.format(model_name))
        missing = [uid for uid in items if uid not in predictions]
        if missing:
            raise ValueError('Model {!r} is missing {} prediction(s), including: {}'.format(
                model_name, len(missing), ', '.join(map(str, missing[:5]))))


def _js_distance(human, model):
    # Preserve the published metric: sqrt(JS divergence), with natural logarithms.
    # KL terms avoid the negative round-off inside scipy's distance implementation.
    midpoint = (human + model) / 2
    divergence = (entropy(human, midpoint) + entropy(model, midpoint)) / 2
    if not np.isfinite(divergence):
        raise ValueError('Could not compute Jensen-Shannon distance')
    return float(np.sqrt(max(0.0, divergence)))


def _mean(values):
    return float(np.mean(values)) if len(values) else float('nan')


def _score_items(items, predictions, task_name):
    if task_name not in ('uncertainty_nli', 'uncertainty_abdnli'):
        raise ValueError('Unknown distributional task: ' + str(task_name))
    size = 3 if task_name == 'uncertainty_nli' else 2
    js_values, kl_values, prediction_entropies = [], [], []
    correct = set()
    old_correct = different = 0
    for uid, item in items.items():
        try:
            human = _vector(item['label_dist'], 'label_dist')
            if len(human) != size or np.any(human < 0) or not np.isclose(human.sum(), 1):
                raise ValueError('label_dist must sum to one and have {} nonnegative values'.format(size))
            prediction = predictions[uid]
            model = _prediction_distribution(prediction, size)
            label = prediction['predicted_label']
            if task_name == 'uncertainty_nli':
                if label not in simply_nli_label_mapping or simply_nli_label_mapping[label] == '-':
                    raise ValueError('Invalid NLI predicted_label: {!r}'.format(label))
                label = simply_nli_label_mapping[label]
            elif label not in (1, 2):
                raise ValueError('alphaNLI predicted_label must be the integer 1 or 2')
            if label == item['majority_label']:
                correct.add(uid)
            old_correct += label == item['old_label']
            different += item['old_label'] != item['majority_label']
            js_values.append(_js_distance(human, model))
            kl_values.append(float(entropy(human, model)))
            prediction_entropies.append(float(entropy(model)))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Invalid example/prediction for UID {!r}: {}'.format(uid, error)) from error
    count = len(items)
    stats = {
        'average JS div': _mean(js_values), 'average KL div': _mean(kl_values),
        'model prediction entropy': _mean(prediction_entropies), 'total_count': count,
        'o_acc': old_correct / count if count else float('nan'),
        'm_acc': len(correct) / count if count else float('nan'),
        'missing_uid': [], 'different_majority_count': different,
        'different_majority_ratio': different / count if count else float('nan'),
    }
    return stats, correct


def calculate_divergence_bwt_model_human_simplify(dist_uid_dict, model_prediction_dict,
                                                task_name='uncertainty_nli'):
    _validate_predictions(dist_uid_dict, model_prediction_dict)
    results = {}
    all_correct = set(dist_uid_dict)
    for name, predictions in model_prediction_dict.items():
        results[name], correct = _score_items(dist_uid_dict, predictions, task_name)
        all_correct.intersection_update(correct)
    for name, predictions in model_prediction_dict.items():
        js_values, kl_values = [], []
        for uid in sorted(all_correct):
            human = np.asarray(dist_uid_dict[uid]['label_dist'], dtype=float)
            model = _prediction_distribution(predictions[uid], len(human))
            js_values.append(_js_distance(human, model))
            kl_values.append(float(entropy(human, model)))
        results[name]['correct example JS div'] = _mean(js_values)
        results[name]['correct example KL div'] = _mean(kl_values)
    return results, all_correct


def build_entropy_bins(dist_uid_dict, num_of_bins, customized_bins=None, type='even'):
    if not dist_uid_dict:
        raise ValueError('Cannot bin an empty dataset')
    if not isinstance(num_of_bins, int) or num_of_bins < 1:
        raise ValueError('num_of_bins must be a positive integer')
    items = list(dist_uid_dict.values())
    entropies = np.asarray([item['entropy'] for item in items], dtype=float)
    if not np.all(np.isfinite(entropies)):
        raise ValueError('Example entropies must be finite')
    if customized_bins is not None:
        bins = np.asarray(customized_bins, dtype=float)
    elif type == 'even':
        _, bins = np.histogram(entropies, bins=num_of_bins)
    elif type == 'quantile':
        bins = np.quantile(entropies, np.linspace(0, 1, num_of_bins + 1))
    else:
        raise ValueError('Unknown bin type: ' + str(type))
    if (bins.shape != (num_of_bins + 1,) or not np.all(np.isfinite(bins))
            or np.any(np.diff(bins) < 0) or bins[0] > entropies.min() or bins[-1] < entropies.max()):
        raise ValueError('Bin boundaries must be ordered and cover all example entropies')
    # Repeated quantiles can produce empty bins. Keep them, with undefined metrics (NaN).
    partitions = []
    for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        selected = [item for value, item in zip(entropies, items)
                    if start <= value and (value <= end if i == num_of_bins - 1 else value < end)]
        partitions.append(((float(start), float(end)), selected))
    return partitions


def calculate_per_bin_results_simplify(bin_partitions, model_prediction_dict, task_name='uncertainty_nli'):
    items = {item['uid']: item for _, partition in bin_partitions for item in partition}
    _validate_predictions(items, model_prediction_dict)
    results = {}
    for name, predictions in model_prediction_dict.items():
        bins = {}
        for bounds, partition in bin_partitions:
            stats, _ = _score_items({item['uid']: item for item in partition}, predictions, task_name)
            stats['range'] = bounds
            # Repeated quantiles can share identical boundaries; they are empty except
            # for the final inclusive bin. Prefer the populated result in that case.
            if bounds not in bins or stats['total_count']:
                bins[bounds] = stats
        results[name] = {'bin_results': bins}
    return results


def build_bin_items(bined_item_results_dict):
    items = []
    for name, result in bined_item_results_dict.items():
        for bounds, stats in result['bin_results'].items():
            items.append({
                'Model': model_name_map_dict.get(name, name),
                'Entropy Range': '{:.3f}-{:.3f}'.format(*bounds),
                'Total Count': stats['total_count'],
                'JSD': round(stats['average JS div'], 4),
                'KL': round(stats['average KL div'], 4),
                'Accuracy': round(stats['m_acc'], 4), 'Old Accuracy': round(stats['o_acc'], 4),
            })
    return pd.DataFrame(items)


def calculate_correlation_btw_model_human_simplify(dist_uid_dict, model_prediction_dict, task_name='unli'):
    from scipy.stats import spearmanr

    if task_name != 'unli':
        raise ValueError('Correlation evaluation requires task_name=unli')
    _validate_predictions(dist_uid_dict, model_prediction_dict)
    human = np.asarray([float(item['unli_label']) for item in dist_uid_dict.values()])
    if not np.all(np.isfinite(human)) or np.any((human < 0) | (human > 1)):
        raise ValueError('UNLI labels must be finite values between zero and one')
    results = {}
    for name, predictions in model_prediction_dict.items():
        model = np.asarray([_prediction_distribution(predictions[uid], 3)[0] for uid in dist_uid_dict])
        if len(human) < 2 or np.ptp(human) == 0 or np.ptp(model) == 0:
            pearson = spearman = float('nan')
        else:
            pearson = float(np.corrcoef(model, human)[0, 1])
            spearman = float(spearmanr(model, human)[0])
        results[name] = {
            'pearson corr': pearson, 'spearman corr': spearman,
            'total_count': len(human), 'missing_uid': [],
        }
    return results
