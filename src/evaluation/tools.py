import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy


def model_label_dist(logits_list):
    logits = np.asarray(logits_list)
    prob = np.exp(logits_list) / np.sum(np.exp(logits_list))
    # numerical stability for KL
    for i, value in enumerate(prob):
        if np.abs(value) < 1e-15:
            prob[i] = 1e-15
    # normalize
    prob = prob / np.sum(prob)
    assert np.isclose(np.sum(prob), 1)
    return prob


def normalize_dist(prob):
    prob = np.asarray(prob)
    for i, value in enumerate(prob):
        if np.abs(value) < 1e-15:
            prob[i] = 1e-15
    # normalize
    prob = prob / np.sum(prob)
    assert np.isclose(np.sum(prob), 1)
    return prob


simply_nli_label_mapping = {
    'e': 'e',
    'c': 'c',
    'n': 'n',
    'entailment': 'e',
    'contradiction': 'c',
    'neutral': 'n',
    '-': '-',
}

simply_nli_label_mapping_model_output_to_original = {
    'e': 'entailment',
    'c': 'contradiction',
    'n': 'neutral',
}

simply_abdnli_label_mapping_model_output_to_original = {
    1: 1,
    2: 2,
}


def calculate_divergence_bwt_model_human_simplify(dist_uid_dict, model_prediction_dict,
                                                  task_name='uncertainty_nli'):
    results_dict = dict()
    all_correct_set = set()

    for model_name, pred_dict in model_prediction_dict.items():
        js_divergence_list = []
        kl_divergence_list = []
        missing_uid = []
        m_acc = 0
        # if original_example_dict is not None:
        original_acc = 0
        total_count = 0
        different_majority_count = 0

        for uid, h_dist_item in dist_uid_dict.items():
            all_correct_set.add(uid)
            # o_uid = reverse_replace_special_mongodb_character(uid)

            if uid not in pred_dict:
                print(f"uid-({uid}) is not in model prediction-({model_name})")
                missing_uid.append(uid)

            # h_label_dist = h_dist_item['label_dist']
            if 'predicted_probabilities' in pred_dict[uid]:
                m_pred_dist = normalize_dist(pred_dict[uid]['predicted_probabilities'])
            else:
                m_pred_dist = model_label_dist(pred_dict[uid]['logits'])  # notice here we're using o_uid.

            human_label_dist = np.asarray(h_dist_item['label_dist'])
            assert np.isclose(np.sum(human_label_dist), 1)
            assert np.isclose(np.sum(m_pred_dist), 1)
            # sanity check

            h_label_dist = h_dist_item['label_dist']
            # m_pred_dist = model_label_dist(pred_dict[uid]['logits'])  # notice here we're using o_uid.

            old_label = h_dist_item['old_label']
            new_label = h_dist_item['majority_label']
            model_p_label = None
            if task_name == 'uncertainty_abdnli':
                model_p_label = pred_dict[uid]['predicted_label']
            elif task_name == 'uncertainty_nli':
                model_p_label = simply_nli_label_mapping[pred_dict[uid]['predicted_label']]
            else:
                raise ValueError()

            if old_label == model_p_label:
                original_acc += 1

            if model_p_label == new_label:
                m_acc += 1
            else:
                all_correct_set.remove(uid)
                # pass

            if old_label != new_label:
                different_majority_count += 1

            # cur_js_divergence = distance.jensenshannon(human_label_dist, m_pred_dist) ** 2
            cur_js_divergence = distance.jensenshannon(human_label_dist, m_pred_dist)
            if np.isnan(cur_js_divergence):
                print("JS for this example is `nan', we will set JS to 0 for the current example. "
                      "This can be a potential error.",
                      "Human distribution:", human_label_dist,
                      "Model distribution:", m_pred_dist,
                      "UID:", uid)
                cur_js_divergence = 0  # set error to 0.
            else:
                pass
                # distance.jensenshannon(human_label_dist, m_pred_dist)
                # print(human_label_dist, m_pred_dist)
            # power 2 to get divergence rather than distance.
            js_divergence_list.append(cur_js_divergence)

            cur_kl_divergence = entropy(human_label_dist, m_pred_dist)
            # think about whether we want to reverse the order or not.
            kl_divergence_list.append(cur_kl_divergence)

            total_count += 1

        avg_js_div = np.mean(js_divergence_list)
        avg_kl_div = np.mean(kl_divergence_list)

        results_dict[model_name] = dict()
        results_dict[model_name]['average JS div'] = avg_js_div
        results_dict[model_name]['average KL div'] = avg_kl_div
        results_dict[model_name]['total_count'] = total_count
        results_dict[model_name]['o_acc'] = original_acc / total_count
        results_dict[model_name]['m_acc'] = m_acc / total_count
        results_dict[model_name]['missing_uid'] = missing_uid
        results_dict[model_name]['different_majority_count'] = different_majority_count
        results_dict[model_name]['different_majority_ratio'] = different_majority_count / total_count

    # calculate all correct subset JS divergence
    for model_name, pred_dict in model_prediction_dict.items():
        correct_js_divergence_list = []
        correct_kl_divergence_list = []

        for uid in all_correct_set:
            h_dist_item = dist_uid_dict[uid]

            if 'predicted_probabilities' in pred_dict[uid]:
                m_pred_dist = normalize_dist(pred_dict[uid]['predicted_probabilities'])
            else:
                m_pred_dist = model_label_dist(pred_dict[uid]['logits'])  # notice here we're using o_uid.

            human_label_dist = np.asarray(h_dist_item['label_dist'])
            assert np.isclose(np.sum(human_label_dist), 1)
            assert np.isclose(np.sum(m_pred_dist), 1)
            # sanity check

            # cur_js_divergence = distance.jensenshannon(human_label_dist, m_pred_dist) ** 2
            cur_js_divergence = distance.jensenshannon(human_label_dist, m_pred_dist)

            # handle potential error that comes with negative js divergence.
            if np.isnan(cur_js_divergence):
                print("JS for this example is `nan', we will set JS to 0 for the current example. "
                      "This can be a potential error.",
                      "Human distribution:", human_label_dist,
                      "Model distribution:", m_pred_dist,
                      "UID:", uid)
                cur_js_divergence = 0   # set error to 0.
            else:
                pass

            # power 2 to get divergence rather than distance.
            correct_js_divergence_list.append(cur_js_divergence)

            cur_kl_divergence = entropy(human_label_dist, m_pred_dist)
            # think about whether we want to reverse the order or not.
            correct_kl_divergence_list.append(cur_kl_divergence)

        avg_js_div = np.mean(correct_js_divergence_list)
        avg_kl_div = np.mean(correct_kl_divergence_list)

        results_dict[model_name]['correct example JS div'] = avg_js_div
        results_dict[model_name]['correct example KL div'] = avg_kl_div

    return results_dict, all_correct_set