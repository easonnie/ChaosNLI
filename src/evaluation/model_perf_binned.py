from evaluation.tools import calculate_divergence_bwt_model_human_simplify, build_entropy_bins, \
    calculate_per_bin_results_simplify, build_bin_items
from utils import common, list_dict_data_tool
import config

import matplotlib.pyplot as plt
import seaborn as sns


def format_number(num, digits=4):
    return str(round(num, digits))


def model_perf_binned(dataset_name, task_name, data_file, model_prediction_file, split_type='quantile', bin_num=5,
                      verbose=True):

    d_list = common.load_jsonl(data_file)
    collected_data_dict = list_dict_data_tool.list_to_dict(d_list, key_fields='uid')
    model_prediction_dict = common.load_json(model_prediction_file)

    bined_item = build_entropy_bins(collected_data_dict, bin_num, type=split_type)
    bined_item_results = calculate_per_bin_results_simplify(bined_item, model_prediction_dict,
                                                            task_name=task_name)

    if verbose:
        print('-' * 60)
        print('Data:', dataset_name)
        for model_name, range_items in bined_item_results.items():
            print('Model: {:20s}'.format(model_name))
            print('\t'.join(['{:18s}'.format('Entropy Range'), '{:15s}'.format('# of Example'),
                             '{:10s}'.format('JSD'), '{:10s}'.format('KL'),
                             '{:10s}'.format('Old Acc.'), '{:10s}'.format('New Acc.')]))
            for range_value, model_item in range_items['bin_results'].items():
                print('\t'.join(['{:5f}-{:5f}'.format(range_value[0], range_value[1]),
                                 '{:15s}'.format(format_number(model_item['total_count'])),
                                 '{:10s}'.format(format_number(model_item['average JS div'])),
                                 '{:10s}'.format(format_number(model_item['average KL div'])),
                                 '{:10s}'.format(format_number(model_item['o_acc'])),
                                 '{:10s}'.format(format_number(model_item['m_acc'])),
                                 ]))
        print('-' * 60)
    return bined_item_results


def plot_histogram(binned_item_results, y_axis_value, column_name):
    model_orders = [
        'BERT-base', 'BERT-large',
        'XLNet-base', 'XLNet-large',
        'RoBERTa-base', 'RoBERTa-large',
        'BART', 'ALBERT',
        'DistilBert',
    ]

    plt.figure(figsize=(12, 2))

    sns.set(style="whitegrid")

    result_items = build_bin_items(binned_item_results)
    ax = sns.barplot(x="Model", y=y_axis_value, hue="Entropy Range",
                     order=model_orders,
                     palette="ch:0.1,-.2,dark=.5",
                     data=result_items)

    if y_axis_value == 'Accuracy':
        ax.set_ylabel(f"Acc. on {column_name}", fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    elif y_axis_value == 'JSD':
        ax.set_ylabel(f"JSD on {column_name}", fontsize=10)
        ax.set_ylim(0.0, 0.5)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    ax.set_xlabel(None)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
    plt.tight_layout()

    # plt.show()

    if y_axis_value == 'Accuracy':
        plt.savefig('abdnli_entropy_bin_perf_acc.pdf')
        # plt.savefig('nli_entropy_bin_perf_acc.pdf')
    elif y_axis_value == 'JSD':
        plt.savefig('abdnli_entropy_bin_perf_jsd.pdf')
        # plt.savefig('nli_entropy_bin_perf_jsd.pdf')


def show_abdnli_binned_plot(y_axis_value):
    dataset_name = 'Abductive Commonsense Reasoning (alphaNLI)'
    task_name = 'uncertainty_abdnli'
    data_file = config.CHAOSNLI_ALPHANLI
    model_pred_file = config.MODEL_PRED_ABDNLI
    column_name = 'ChaosNLI-$\\alpha$'

    bined_item_results = model_perf_binned(dataset_name, task_name, data_file, model_pred_file, verbose=False)
    plot_histogram(bined_item_results, y_axis_value, column_name)


def show_nli_binned_plot(y_axis_value):
    dataset_name = 'Natural Language Inference'
    task_name = 'uncertainty_nli'
    snli_data_file = config.CHAOSNLI_SNLI
    mnli_data_file = config.CHAOSNLI_MNLI

    model_pred_file = config.MODEL_PRED_NLI

    d_list_snli = common.load_jsonl(snli_data_file)
    d_list_mnli = common.load_jsonl(mnli_data_file)

    collected_data_dict = {}
    collected_data_dict_snli = list_dict_data_tool.list_to_dict(d_list_snli, key_fields='uid')
    collected_data_dict_mnli = list_dict_data_tool.list_to_dict(d_list_mnli, key_fields='uid')
    collected_data_dict.update(collected_data_dict_snli)
    collected_data_dict.update(collected_data_dict_mnli)

    model_prediction_dict = common.load_json(model_pred_file)

    bin_num = 5
    split_type = 'quantile'
    column_name = 'ChaosNLI-(S+M)'

    bined_item = build_entropy_bins(collected_data_dict, bin_num, type=split_type)
    bined_item_results = calculate_per_bin_results_simplify(bined_item, model_prediction_dict,
                                                            task_name=task_name)

    plot_histogram(bined_item_results, y_axis_value, column_name)


def model_perf_snli_binned():
    dataset_name = 'ChaosNLI - Stanford Natural Language Inference (SNLI)'
    task_name = 'uncertainty_nli'
    data_file = config.CHAOSNLI_SNLI
    model_pred_file = config.MODEL_PRED_NLI

    model_perf_binned(dataset_name, task_name, data_file, model_pred_file)


def model_perf_mnli_binned():
    dataset_name = 'ChaosNLI - Multi-Genre Natural Language Inference (MNLI)'
    task_name = 'uncertainty_nli'
    data_file = config.CHAOSNLI_MNLI
    model_pred_file = config.MODEL_PRED_NLI

    model_perf_binned(dataset_name, task_name, data_file, model_pred_file)


def model_perf_abdnli_binned():
    dataset_name = 'ChaosNLI - Abductive Commonsense Reasoning (alphaNLI)'
    task_name = 'uncertainty_abdnli'
    data_file = config.CHAOSNLI_ALPHANLI
    model_pred_file = config.MODEL_PRED_ABDNLI

    model_perf_binned(dataset_name, task_name, data_file, model_pred_file)


if __name__ == '__main__':
    model_perf_abdnli_binned()
    model_perf_snli_binned()
    model_perf_mnli_binned()

    # show_abdnli_binned_plot('Accuracy')
    # show_abdnli_binned_plot('JSD')
    # show_nli_binned_plot('Accuracy')
    # show_nli_binned_plot('JSD')
