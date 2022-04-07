import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy, spearmanr
import pandas as pd


def format_number(num, digits=4):
    return str(round(num, digits))


model_name_map_dict = {
    'bert-base': 'BERT-base',
    'bert-large': 'BERT-large',
    'xlnet-base': 'XLNet-base',
    'xlnet-large': 'XLNet-large',
    'roberta-base': 'RoBERTa-base',
    'roberta-large': 'RoBERTa-large',
    'distilbert': 'DistilBert',
    'albert-xxlarge': 'ALBERT',
    'bart-large': 'BART',
}


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


def build_bin_items(bined_item_results_dict):
    all_items = []
    # write a Model name converter
    for model_name, range_items in bined_item_results_dict.items():
        for range_value, model_item in range_items['bin_results'].items():
            new_item = dict()

            new_item['Model'] = model_name_map_dict[model_name]
            # new_item['Model'] = model_name
            new_item['Entropy Range'] = '{:.3f}-{:.3f}'.format(range_value[0], range_value[1])
            new_item['Total Count'] = int(format_number(model_item['total_count']))
            new_item['JSD'] = float(format_number(model_item['average JS div']))
            new_item['KL'] = float(format_number(model_item['average KL div']))
            new_item['Accuracy'] = float(format_number(model_item['m_acc']))
            new_item['Old Accuracy'] = float(format_number(model_item['o_acc']))
            all_items.append(new_item)

    return pd.DataFrame(all_items)


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


def build_entropy_bins(dist_uid_dict, num_of_bins, customized_bins=None, type='even'):
    # output format:
    # list of tuple, first indicate the entropy range, the later is the items included in the ranage.
    entropy_list = []
    items_list = []
    for uid, item in dist_uid_dict.items():
        entropy_list.append(item['entropy'])
        items_list.append(item)

    assert len(entropy_list) == len(items_list)

    if customized_bins is not None:
        bins = customized_bins
    else:
        if type == 'even':
            out, bins = np.histogram(entropy_list, bins=num_of_bins)
        elif type == 'quantile':
            nquent = np.linspace(0, 1, num_of_bins + 1)
            bins = []
            bins.append(min(entropy_list))  # start index is the min
            for i in range(num_of_bins):
                bins.append(np.quantile(entropy_list, nquent[i + 1]))

    assert len(bins) == num_of_bins + 1

    partitioned_range = []
    partitioned_examples = []
    returned_partition = []

    for _ in range(num_of_bins):
        partitioned_examples.append([])

    for i in range(num_of_bins):
        vstart = bins[i]
        vend = bins[i + 1]
        partitioned_range.append((vstart, vend))
        for cur_entropy, cur_item in zip(entropy_list, items_list):
            if i == num_of_bins - 1:
                if vstart <= cur_entropy <= vend:
                    partitioned_examples[i].append(cur_item)
            else:
                if vstart <= cur_entropy < vend:  # last bin is close on end
                    partitioned_examples[i].append(cur_item)

        return_bins_item = ((vstart, vend), partitioned_examples[i])
        returned_partition.append(return_bins_item)

    # check mutually exclusive and sum to one.
    assert len(entropy_list) == sum([len(bin_examples) for bin_examples in partitioned_examples])

    return returned_partition

    # indices = np.digitize(entropy_list, bins, right=True)
    # print(indices)
    # print(Counter(indices))
    # ind_max = max(indices)
    # ind_min = min(indices)
    # num_of_bin = ind_max - ind_min
    # print(num_of_bin)

def calculate_per_bin_results_simplify(bin_partitions, model_prediction_dict, task_name='uncertainty_nli'):
    results_dict = dict()

    if task_name == 'uncertainty_nli':
        pass
    elif task_name == 'uncertainty_abdnli':
        pass

    for model_name, pred_dict in model_prediction_dict.items():
        results_dict[model_name] = dict()
        results_dict[model_name]['bin_results'] = dict()
        all_count = 0
        for (vstart, vend), bin_items in bin_partitions:
            results_dict[model_name]['bin_results'][(vstart, vend)] = dict()

            js_divergence_list = []
            kl_divergence_list = []
            model_prediction_entropy_list = []
            missing_uid = []
            m_acc = 0
            original_acc = 0
            total_count = 0
            different_majority_count = 0

            for h_dist_item in bin_items:
                uid = h_dist_item['uid']
                # o_uid = reverse_replace_special_mongodb_character(uid)
                #
                if uid not in pred_dict:
                    print(f"uid-({uid}) is not in model prediction-({model_name})")
                    missing_uid.append(uid)

                human_label_dist = np.asarray(h_dist_item['label_dist'])
                h_label_dist = h_dist_item['label_dist']

                if 'predicted_probabilities' in pred_dict[uid]:
                    m_pred_dist = normalize_dist(pred_dict[uid]['predicted_probabilities'])
                else:
                    m_pred_dist = model_label_dist(pred_dict[uid]['logits'])  # notice here we're using o_uid.

                assert np.isclose(np.sum(human_label_dist), 1)
                assert np.isclose(np.sum(m_pred_dist), 1)
                # sanity check

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
                    pass
                    # all_correct_set.remove(uid)
                    # pass

                if old_label != new_label:
                    different_majority_count += 1

                # cur_js_divergence = distance.jensenshannon(human_label_dist, m_pred_dist) ** 2
                cur_js_divergence = distance.jensenshannon(human_label_dist, m_pred_dist)
                # power 2 to get divergence rather than distance.

                if np.isnan(cur_js_divergence):
                    print("JS for this example is `nan', we will set JS to 0 for the current example. "
                          "This can be a potential error.",
                          "Human distribution:", human_label_dist,
                          "Model distribution:", m_pred_dist,
                          "UID:", uid)
                    cur_js_divergence = 0  # set error to 0.
                else:
                    pass

                js_divergence_list.append(cur_js_divergence)

                cur_kl_divergence = entropy(human_label_dist, m_pred_dist)
                # think about whether we want to reverse the order or not.
                kl_divergence_list.append(cur_kl_divergence)

                model_prediction_entropy_list.append(entropy(m_pred_dist))

                total_count += 1
                all_count += 1

            avg_js_div = np.mean(js_divergence_list)
            avg_kl_div = np.mean(kl_divergence_list)

            model_prediction_avg_entropy = np.mean(model_prediction_entropy_list)

            results_dict[model_name]['bin_results'][(vstart, vend)]['range'] = (vstart, vend)
            results_dict[model_name]['bin_results'][(vstart, vend)]['total_count'] = total_count
            results_dict[model_name]['bin_results'][(vstart, vend)]['average JS div'] = avg_js_div
            results_dict[model_name]['bin_results'][(vstart, vend)]['average KL div'] = avg_kl_div
            results_dict[model_name]['bin_results'][(vstart, vend)]['model prediction entropy'] = model_prediction_avg_entropy
            results_dict[model_name]['bin_results'][(vstart, vend)]['total_count'] = total_count
            results_dict[model_name]['bin_results'][(vstart, vend)]['o_acc'] = original_acc / total_count
            results_dict[model_name]['bin_results'][(vstart, vend)]['m_acc'] = m_acc / total_count
            results_dict[model_name]['bin_results'][(vstart, vend)]['missing_uid'] = missing_uid
            results_dict[model_name]['bin_results'][(vstart, vend)][
                'different_majority_count'] = different_majority_count
            results_dict[model_name]['bin_results'][(vstart, vend)][
                'different_majority_ratio'] = different_majority_count / total_count

        # print("Processed Bin Item Count:", all_count)

    return results_dict


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




def calculate_correlation_btw_model_human_simplify(dist_uid_dict, model_prediction_dict,
                                                  task_name='unli'):

    assert(task_name=='unli')

    results_dict = dict()

    for model_name, pred_dict in model_prediction_dict.items():
        js_divergence_list = []
        kl_divergence_list = []
        missing_uid = []
        entail_pred_list = []
        human_unli_label_list = []
        m_acc = 0
        # if original_example_dict is not None:
        original_acc = 0
        total_count = 0
        different_majority_count = 0

        for uid, h_dist_item in dist_uid_dict.items():
            # o_uid = reverse_replace_special_mongodb_character(uid)

            if uid not in pred_dict:
                print(f"uid-({uid}) is not in model prediction-({model_name})")
                missing_uid.append(uid)

            # h_label_dist = h_dist_item['label_dist']
            if 'predicted_probabilities' in pred_dict[uid]:
                m_pred_dist = normalize_dist(pred_dict[uid]['predicted_probabilities'])
            else:
                m_pred_dist = model_label_dist(pred_dict[uid]['logits'])  # notice here we're using o_uid.


            entail_pred_dist = m_pred_dist[0]

            human_unli_label = np.asarray(float(h_dist_item['unli_label']))
            assert np.isclose(np.sum(m_pred_dist), 1)
            # sanity check



            entail_pred_list.append(entail_pred_dist)
            human_unli_label_list.append(human_unli_label)


            total_count += 1

        entail_pred_list = np.asarray(entail_pred_list).flatten()
        human_unli_label_list = np.asarray(human_unli_label_list).flatten()
        # print(entail_pred_list[0])
        # print(human_unli_label_list[0])
        # print(type(entail_pred_list[0]))
        # print(type(human_unli_label_list[0]))
        # print(len(entail_pred_list))
        # print(len(human_unli_label_list))


        pearson_corr = np.corrcoef(entail_pred_list, human_unli_label_list)[0][1]
        spearman_corr = spearmanr(entail_pred_list, human_unli_label_list)[0]

        results_dict[model_name] = dict()
        results_dict[model_name]['pearson corr'] = pearson_corr
        results_dict[model_name]['spearman corr'] = spearman_corr
        results_dict[model_name]['total_count'] = total_count
        results_dict[model_name]['missing_uid'] = missing_uid

    return results_dict
