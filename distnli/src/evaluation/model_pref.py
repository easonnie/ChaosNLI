from evaluation.tools import calculate_divergence_bwt_model_human_simplify, calculate_correlation_btw_model_human_simplify
from utils import common, list_dict_data_tool
import config


def format_number(num, digits=4):
    return str(round(num, digits))


def model_perf_unli(dataset_name, task_name, data_file, model_prediction_file):
    d_list = common.load_jsonl(data_file)
    collected_data_dict = list_dict_data_tool.list_to_dict(d_list, key_fields='uid')
    model_prediction_dict = common.load_json(model_prediction_file)
    results_dict = calculate_correlation_btw_model_human_simplify(collected_data_dict,
                                                                                  model_prediction_dict, task_name)
    print('-' * 60)
    print('Data:', dataset_name)
    print('\t'.join(['{:20s}'.format('Model Name'), '{:10s}'.format('Pearson Corr'), '{:10s}'.format('Spearman Corr')]))
    for model_name, model_item in results_dict.items():
        print('\t'.join(['{:20s}'.format(model_name),
                         '{:10s}'.format(format_number(model_item['pearson corr'])),
                         '{:10s}'.format(format_number(model_item['spearman corr'])),
                         ]))
    print('-' * 60)


def model_perf(dataset_name, task_name, data_file, model_prediction_file):
    d_list = common.load_jsonl(data_file)
    collected_data_dict = list_dict_data_tool.list_to_dict(d_list, key_fields='uid')
    model_prediction_dict = common.load_json(model_prediction_file)
    results_dict, all_correct_set = calculate_divergence_bwt_model_human_simplify(collected_data_dict,
                                                                                  model_prediction_dict, task_name)
    print('-' * 60)
    print('Data:', dataset_name)
    print("All Correct Count:", len(all_correct_set))
    print('\t'.join(['{:20s}'.format('Model Name'), '{:10s}'.format('JSD'), '{:10s}'.format('KL'),
                     '{:10s}'.format('Old Acc.'), '{:10s}'.format('New Acc.')]))
    for model_name, model_item in results_dict.items():
        print('\t'.join(['{:20s}'.format(model_name),
                         '{:10s}'.format(format_number(model_item['average JS div'])),
                         '{:10s}'.format(format_number(model_item['average KL div'])),
                         '{:10s}'.format(format_number(model_item['o_acc'])),
                         '{:10s}'.format(format_number(model_item['m_acc'])),
                         ]))
    print('-' * 60)


def model_perf_snli():
    dataset_name = 'ChaosNLI - Stanford Natural Language Inference (SNLI)'
    task_name = 'uncertainty_nli'
    data_file = config.CHAOSNLI_SNLI
    model_pred_file = config.MODEL_PRED_NLI

    model_perf(dataset_name, task_name, data_file, model_pred_file)


def model_perf_mnli():
    dataset_name = 'ChaosNLI - Multi-Genre Natural Language Inference (MNLI)'
    task_name = 'uncertainty_nli'
    data_file = config.CHAOSNLI_MNLI
    model_pred_file = config.MODEL_PRED_NLI

    model_perf(dataset_name, task_name, data_file, model_pred_file)


def model_perf_abdnli():
    dataset_name = 'ChaosNLI - Abductive Commonsense Reasoning (alphaNLI)'
    task_name = 'uncertainty_abdnli'
    data_file = config.CHAOSNLI_ALPHANLI
    model_pred_file = config.MODEL_PRED_ABDNLI

    model_perf(dataset_name, task_name, data_file, model_pred_file)


if __name__ == '__main__':
    model_perf_abdnli()
    model_perf_snli()
    model_perf_mnli()
