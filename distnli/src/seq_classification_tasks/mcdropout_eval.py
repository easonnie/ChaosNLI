import argparse
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
import config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from flint.data_utils.batchbuilder import BaseBatchBuilder, move_to_device
from flint.data_utils.fields import RawFlintField, LabelFlintField, ArrayIndexFlintField
from seq_classification_tasks.data_converter import nli

# from modeling.multichoice_modeling import MultichoiceModel
from modeling.registrated_models import SEQ_CLF_MODEL_CLASSES
from seq_classification_tasks.data_loader import SeqClassificationTransform, ListDataSet
from utils import common, list_dict_data_tool, save_tool
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import random
import torch
from tqdm import tqdm

import pprint

pp = pprint.PrettyPrinter(indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="If set, we only use CPU.")
    parser.add_argument("--single_gpu", action="store_true", help="If set, we only use single GPU.")
    parser.add_argument("--fp16", action="store_true", help="If set, we will use fp16.")

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    # environment arguments
    parser.add_argument('-s', '--seed', default=1, type=int, metavar='N',
                        help='manual random seed')
    parser.add_argument('-n', '--num_nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--node_rank', default=0, type=int,
                        help='ranking within the nodes')

    # experiments specific arguments
    parser.add_argument('--debug_mode',
                        action='store_true',
                        dest='debug_mode',
                        help='weather this is debug mode or normal')

    parser.add_argument(
        "--model_class_name",
        type=str,
        help="Set the model class of the experiment.",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Set the name of the experiment. [model_name]/[data]/[task]/[other]",
    )

    parser.add_argument(
        "--load_model_path",
        type=str,
        help="path of the state_dict of the model to load",
    )

    parser.add_argument(
        "--save_prediction",
        action='store_true',
        dest='save_prediction',
        help='Do we want to save prediction')
    parser.add_argument(
        "--save_checkpoint",
        action='store_true',
        dest='save_checkpoint',
        help='Do we want to save checkpoints')

    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--number_of_choices', default=2, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--total_step', default=-1, type=int, metavar='N',
                        help='number of step to update, default calculate with total data size.'
                             'if we set this step, then epochs will be 100 to run forever.')

    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument("--max_length", default=160, type=int, help="Max length of the sequences.")

    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument(
        "--eval_frequency", default=1000, type=int, help="set the evaluation frequency, evaluate every X global step.",
    )

    parser.add_argument("--snli_eval_dataset_name", type=str, default=None, help='the name tag for the training dataset')
    parser.add_argument("--mnli_eval_dataset_name", type=str, default=None, help='the name tag for the training dataset')
    parser.add_argument('--mcdropout_pass_num', type=int, default=10, help='number of passes for mcdropout')
    args = parser.parse_args()

    if args.cpu:
        args.world_size = 1
        start(-1, args)
    elif args.single_gpu:
        args.world_size = 1
        start(0, args)
    else:  # distributed multiGPU training
        #########################################################
        args.world_size = args.gpus_per_node * args.num_nodes  #
        os.environ['MASTER_ADDR'] = 'localhost'
        # maybe we will automatically retrieve the IP later.
        os.environ['MASTER_PORT'] = '88888'  #
        mp.spawn(start, nprocs=args.gpus_per_node, args=(args,))  # spawn how many process in this node
        # remember train is called as train(i, args).
        #########################################################


def start(local_rank, args):
    # debug = False
    # print("GPU:", gpu)
    # world_size = args.world_size
    args.global_rank = args.node_rank * args.gpus_per_node + local_rank
    args.local_rank = local_rank
    # args.warmup_steps = 20
    debug_count = 1000


    set_seed(args.seed)
    # num_labels = 1  # remember here is 1

    max_length = args.max_length

    number_of_classification_label = 3  # NLI: 3

    model_class_item = SEQ_CLF_MODEL_CLASSES[args.model_class_name]
    model_name = model_class_item['model_name']
    do_lower_case = model_class_item['do_lower_case'] if 'do_lower_case' in model_class_item else False

    tokenizer = model_class_item['tokenizer'].from_pretrained(model_name,
                                                              cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                              do_lower_case=do_lower_case)

    encoder_model = model_class_item['sequence_classification'].from_pretrained(model_name,
                                                                                cache_dir=str(
                                                                                    config.PRO_ROOT / "trans_cache"),
                                                                                num_labels=number_of_classification_label)

    # model = MultichoiceModel(encoder_model, args.number_of_choices)
    model = encoder_model

    model.load_state_dict(torch.load(args.load_model_path), strict=False)

    padding_token_value = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_segement_value = model_class_item["padding_segement_value"]
    padding_att_value = model_class_item["padding_att_value"]
    left_pad = model_class_item['left_pad'] if 'left_pad' in model_class_item else False

    batch_size_per_gpu_eval = args.per_gpu_eval_batch_size

    if not args.cpu and not args.single_gpu:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.global_rank
        )


    task_name = 'NLI'

#    dev_snli_list = nli.get_data('snli', 'dev')
#    mnli_dev_m_list = nli.get_data('mnli', 'dev_m')

    if args.snli_eval_dataset_name is not None:
        dev_snli_list = nli.get_data('snli', args.snli_eval_dataset_name)
    else:
        dev_snli_list = nli.get_data('snli', 'dev')
    if args.mnli_eval_dataset_name is not None:
        mnli_dev_m_list = nli.get_data('mnli', args.mnli_eval_dataset_name)
    else:
        mnli_dev_m_list = nli.get_data('mnli', 'dev_m')


    # test_list = alphaNLI.get_data('test')


    data_transformer = SeqClassificationTransform(model_name, tokenizer, max_length)

    snli_dev_dataset = ListDataSet(dev_snli_list, data_transformer)
    mnli_dev_m_dataset = ListDataSet(mnli_dev_m_list, data_transformer)

    snli_dev_sampler = SequentialSampler(snli_dev_dataset)
    mnli_dev_m_sampler = SequentialSampler(mnli_dev_m_dataset)


    batching_schema = data_transformer.get_batching_schema(padding_token_value,
                                                           padding_segement_value,
                                                           padding_att_value, left_pad)


    snli_dev_dataloader = DataLoader(dataset=snli_dev_dataset,
                                     batch_size=batch_size_per_gpu_eval,
                                     shuffle=False,  #
                                     num_workers=0,
                                     pin_memory=True,
                                     sampler=snli_dev_sampler,
                                     collate_fn=BaseBatchBuilder(batching_schema))  #

    mnli_dev_m_dataloader = DataLoader(dataset=mnli_dev_m_dataset,
                                       batch_size=batch_size_per_gpu_eval,
                                       shuffle=False,  #
                                       num_workers=0,
                                       pin_memory=True,
                                       sampler=mnli_dev_m_sampler,
                                       collate_fn=BaseBatchBuilder(batching_schema))  #

    if not args.cpu:
        torch.cuda.set_device(args.local_rank)
        model.cuda(args.local_rank)

    #if args.fp16:
    #    try:
    #        from apex import amp
    #    except ImportError:
    #        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if not args.cpu and not args.single_gpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    output_device=local_rank, find_unused_parameters=True)

    args_dict = dict(vars(args))
    file_path_prefix = '.'

    is_finished = False

    # Let build the logger and log everything before the start of the first training epoch.
    if args.global_rank in [-1, 0]:  # only do logging if we use cpu or global_rank=0
        if not args.debug_mode:
            file_path_prefix, date = save_tool.gen_file_prefix(f"{args.experiment_name}")
            # # # Create Log File
            # Save the source code.
            script_name = os.path.basename(__file__)
            with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
                out_f.write(it.read())
                out_f.flush()

            # Save option file
            common.save_json(args_dict, os.path.join(file_path_prefix, "args.json"))
            checkpoints_path = Path(file_path_prefix) / "checkpoints"
            if not checkpoints_path.exists():
                checkpoints_path.mkdir()
            prediction_path = Path(file_path_prefix) / "mtdropout_predictions"
            if not prediction_path.exists():
                prediction_path.mkdir()


    # print(f"Global Rank:{args.global_rank} ### ", 'Init!')

    if args.global_rank in [-1, 0]:
        r_dict = dict()
        pred_output_list = eval_model_mtdropout(model, snli_dev_dataloader, args.global_rank, args, T=args.mcdropout_pass_num)
        hit, total = count_acc(dev_snli_list, pred_output_list)
        # print(hit, total)
        # exit(0)

        print(debug_node_info(args), f"SNLI Dev Acc:", hit, total, hit / total)
        r_dict[f'snli_dev'] = {
            'acc': hit / total,
            'correct_count': hit,
            'total_count': total,
            'predictions': pred_output_list,
        }

        # saving checkpoints

        pred_output_list = eval_model_mtdropout(model, mnli_dev_m_dataloader, args.global_rank, args, T=args.mcdropout_pass_num)
        hit, total = count_acc(mnli_dev_m_list, pred_output_list)
        # print(hit, total)
        # exit(0)

        print(debug_node_info(args), f"MNLI Dev M Acc:", hit, total, hit / total)
        r_dict[f'mnli_dev_m'] = {
            'acc': hit / total,
            'correct_count': hit,
            'total_count': total,
            'predictions': pred_output_list,
        }

        # saving checkpoints
        snli_dev_acc = r_dict[f'snli_dev']['acc']
        mnli_dev_m_acc = r_dict[f'mnli_dev_m']['acc']

        current_checkpoint_filename = \
            f'|snli_dev({round(snli_dev_acc, 4)})' \
            f'|mnli_dev_m({round(mnli_dev_m_acc, 4)})'

        # save prediction:
        if not args.debug_mode and args.save_prediction:
            cur_results_path = prediction_path / current_checkpoint_filename
            if not cur_results_path.exists():
                cur_results_path.mkdir(parents=True)
            for key, item in r_dict.items():
                common.save_jsonl(item['predictions'], cur_results_path / f"{key}.jsonl")



def count_acc(gt_list, pred_list):
    assert len(gt_list) == len(pred_list)
    gt_dict = list_dict_data_tool.list_to_dict(gt_list, 'uid')
    pred_list = list_dict_data_tool.list_to_dict(pred_list, 'uid')
    total_count = 0
    hit = 0

    id2label = {
        0: 'e',
        1: 'n',
        2: 'c',
        -1: '-',
    }

    for key, value in pred_list.items():
        if gt_dict[key]['label'] == id2label[value['predicted_label']]:
            hit += 1
        total_count += 1
    return hit, total_count


def eval_model_mtdropout(model, dev_dataloader, device_num, args, T=10):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    uid_list = []
    y_list = []
    pred_list = []
    logits_list = []

    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader, 0):
            batch = move_to_device(batch, device_num)
            
            all_probs = []
            for j in range(T):
                if args.model_class_name in ["distilbert", "bart-large"]:
                    outputs = model(batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    labels=None)
                else:
                    outputs = model(batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    token_type_ids=batch['token_type_ids'],
                                    labels=None)

                logits = outputs[0]
                probabilities = F.softmax(logits, -1)
                all_probs.append(probabilities)
            mean_prob = torch.mean(torch.stack(all_probs), 0)
            log_mean_prob = torch.log(mean_prob)

            # loss, logits = outputs[:2]
            uid_list.extend(list(batch['uid']))
            y_list.extend(batch['y'].tolist())
            pred_list.extend(torch.max(log_mean_prob, 1)[1].view(log_mean_prob.size(0)).tolist())
            logits_list.extend(log_mean_prob.tolist())

    assert len(pred_list) == len(logits_list)
    assert len(pred_list) == len(logits_list)

    result_items_list = []
    for i in range(len(uid_list)):
        r_item = dict()
        r_item['uid'] = uid_list[i]
        r_item['logits'] = logits_list[i]
        r_item['predicted_label'] = pred_list[i]

        result_items_list.append(r_item)

    return result_items_list



def debug_node_info(args):
    names = ['global_rank', 'local_rank', 'node_rank']
    values = []

    for name in names:
        if name in args:
            values.append(getattr(args, name))
        else:
            return "Pro:No node info "

    return "Pro:" + '|'.join([f"{name}:{value}" for name, value in zip(names, values)]) + "||Print:"


if __name__ == '__main__':
    main()
