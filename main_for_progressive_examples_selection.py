# 多了能够在选择完最useful的例子后，在top useful的例子里随机选择一部分来在train上验证的结果。


import copy
import os
import argparse
import pickle as pkl
import random
import torch
import math
import logging

logger = logging.getLogger(__name__)
import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer,AutoModelForCausalLM

from data import load_data, prepare_data
from run import train, inference, my_inference
from model_util import load_checkpoint, set_extra_embeddings, \
    set_separate_lm_head, set_separate_embeddings, set_transformed_lm_head
from util import get_prompts, get_paths, flatten_label_losses, \
    prepend_task_tokens, reassign_output_tokens

import fitlog
import csv
import json
from my_dataset import MyDataset

# fitlog.debug()

# fitlog.set_log_dir('fitlog_dir_selection_train')

N_LABELS_DICT = {"SST-2": 2, "sst-5": 5, "mr": 2, "cr": 2, "mpqa": 2,
                 "subj": 2, "trec": 6, "CoLA": 2,
                 "amazon": 5, "yelp_full": 5, "yelp_binary": 2,
                 "agnews": 4, "copa": 2, "boolq": 2,
                 "RTE": 2, "cb": 3,
                 "yahoo": 10, "dbpedia": 14, 'snli': 3, 'mnli': 3, 'qnli': 2, 'wnli': 2, 'amazon_b': 2}

full_data_dir_dict = {'SST-2': 'full_train_data/SST-2',
                      'sst-5': 'full_train_data/sst-5',
                      # 'dbpedia':'../data_full/original/TextClassificationDatasets/dbpedia_csv',
                      'subj': 'full_train_data/subj',
                      'trec': 'full_train_data/trec',
                      'mr': 'full_train_data/mr',
                      'agnews': 'full_train_data/agnews',
                      'amazon_b': 'full_train_data/amazon_b',
                      'dbpedia': 'full_train_data/dbpedia',
                      }


def load_data_by_fp(task, data_fp=None, split='train'):
    if data_fp == None:
        if task == 'SST-2':
            data_fp = '{}/train.tsv'.format(full_data_dir_dict['SST-2'])
        elif task == 'sst-5':
            data_fp = '{}/train.csv'.format(full_data_dir_dict['sst-5'])
        elif task == 'subj':
            data_fp = '{}/train.csv'.format(full_data_dir_dict['subj'])
        elif task == 'trec':
            data_fp = '{}/train.csv'.format(full_data_dir_dict['trec'])
        elif task == 'agnews':
            data_fp = '{}/train_30000.csv'.format(full_data_dir_dict['agnews'])
        elif task == 'mr':
            data_fp = '{}/train.csv'.format(full_data_dir_dict['mr'])
        elif task == 'dbpedia':
            data_fp = '{}/train_30000.csv'.format(full_data_dir_dict['dbpedia'])
        elif task == 'amazon_b':
            data_fp = '{}/train_30000.csv'.format(full_data_dir_dict['amazon_b'])
        else:
            raise NotImplementedError

        # assert split in ['train','test']
        assert split == 'train'

        # if split=='test':
        #     if task in ['agnews','yelp_full','yahoo','dbpedia','amazon']:
        #         data_fp = '{}/test.csv'.format(full_data_dir_dict[task])
        #     else:
        #         data_fp = data_fp.replace('train','test')

    data = []
    if os.path.exists(data_fp):
        if data_fp.endswith('.tsv'):
            if task in ['qnli']:
                with open(data_fp) as f:
                    for line in f:
                        s, q, label = line.strip().split('\t')
                        data.append([
                            'Premise: {} Question:{}'.format(s, q),
                            label
                        ])
            elif task in ['wnli', 'snli', 'mnli']:
                with open(data_fp) as f:
                    for line in f:
                        s1, s2, label = line.strip().split('\t')
                        data.append([
                            'Premise: {} Hypothesis: {}'.format(s1, s2),
                            label
                        ])
            else:
                with open(data_fp) as f:
                    for line in f:
                        data.append(line.strip().split("\t"))
        elif data_fp.endswith('.csv'):
            with open(data_fp) as f:
                if task in ['agnews', 'amazon', 'dbpedia', 'amazon_b']:
                    for label, title, text in csv.reader(f):
                        inp_sentence = title.strip() + '. ' + text.strip()
                        data.append([inp_sentence, label])

                    a = set(map(lambda x: int(x[1]), data))
                    if min(a) == 1:
                        data = list(map(lambda x: [x[0],
                                                   str(int(x[1]) - 1)], data
                                        ))
                elif task == 'yahoo':
                    for label, text1, text2, text3 in csv.reader(f):
                        text3 = text3.replace("\t", " ").replace("\\n", " ")
                        inp_sentence = ' '.join([text1, text2, text3])
                        data.append([inp_sentence, label])

                    a = set(map(lambda x: int(x[1]), data))
                    logger.info('min(a)={}'.format(min(a)))
                    if min(a) == 1:
                        data = list(map(lambda x: [x[0],
                                                    str(int(x[1]) - 1)], data
                                        ))

                else:
                    for label, text in csv.reader(f):
                        data.append([text, label])
                    a = set(map(lambda x: int(x[1]), data))
                    if min(a) == 1:
                        data = list(map(lambda x: [x[0],
                                                   str(int(x[1]) - 1)], data
                                        ))

        if task == "CoLA":
            data = [(sent, label) for _, label, _, sent in data]
        elif task == "RTE":
            data = [(json.dumps({
                "text": p, "question": h[:-1] if h.endswith(".") else h
            }), "1" if l == "entailment" else "0")
                for _, p, h, l in data[1:]]
        elif data[0] == ["sentence", "label"]:
            data = data[1:]
    else:
        logger.info('{} does not exist.'.format(data_fp))
        raise NotImplementedError

    for i in range(5):
        print(data[i])

    for i, dp in enumerate(data):
        dp[0] = ' '.join(dp[0].strip().split())

    # all data should have (input, output) format
    print('data[0] in text_dataset: {}'.format(data[0]))
    assert np.all([len(dp) == 2 for dp in data])

    return data


def get_candidate_indication_pair_loss(args, task, tokenizer, model,
                                       candidate_data, indication_data,
                                       max_length, template_idx, cache_path):
    n_gpu = torch.cuda.device_count()
    n_classes = N_LABELS_DICT.get(task, None)
    templates = get_prompts(task, template_idx)

    # n_classes = N_LABELS_DICT[args.task]

    # for debug
    # candidate_num = len(candidate_data)
    # indication_num = len(indication_data)
    #
    # for_debug_result = torch.rand(size=[candidate_num + 1,indication_num, n_classes])
    #
    # return for_debug_result

    max_length_per_example = max_length
    mydataset = MyDataset(args, candidate_data, indication_data,
                          args.method, tokenizer,
                          max_length_per_example=max_length_per_example,
                          n_classes=N_LABELS_DICT[args.task], template=templates, add_zero_shot_pseudo_candidate=1,
                          candidate_demonstrations_element_check_tag='example')
    logger.info('mydataset size:{}'.format(len(mydataset)))

    logger.info("Checking the first example...")
    input_ids = mydataset[0]["input_ids"].numpy().tolist()
    token_type_ids = mydataset[0]["token_type_ids"].numpy().tolist()
    logger.info("Input:")
    logger.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
    logger.info("Output:")
    logger.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))

    if cache_path != None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            loss_matrix = pkl.load(f)

    else:
        model.eval()

        loss_matrix = my_inference(model, mydataset, args.batch_size)

        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pkl.dump(loss_matrix, f)

    return loss_matrix


def transform_loss_matrix_into_acc(loss_matrix, indication_y_s):
    # loss_matrix [candidate_num, indication_num, class_num]
    # indication_y_s [indication_num]
    # return [candidate_num, indication_num]
    prediction_classes = torch.argmin(loss_matrix, dim=-1)
    if type(indication_y_s) != torch.Tensor:
        indication_y_s = torch.Tensor(indication_y_s)

    indication_y_s = torch.unsqueeze(indication_y_s, dim=0)
    result = (indication_y_s == prediction_classes)
    return result
    pass


def transform_loss_matrix_into_final_loss(loss_matrix, indication_y_s, args, use_true_p):
    # loss_matrix [candidate_num, indication_num, class_num]
    # indication_y_s [indication_num]
    # use_true_p: loss_matrix本身是-log p，如果直接用它来算最终的p，实际作用可能差不多，但公式对不上，如果这个为真，就用exp(-loss_matrix)还原回原来的p
    # channel_p_re_scale: channel计算loss的时候是取indication的x的所有token的loss的平均，如果要得到channel的真p，就得再loss_matrix基础上*n，n为每个indication的token的数量
    # indication_x_s 是为了在

    assert not use_true_p
    # assert not channel_p_re_scale

    candidate_num = loss_matrix.size(0)
    indication_num = loss_matrix.size(1)
    class_num = loss_matrix.size(2)

    loss_matrix_all_label_sum = torch.sum(loss_matrix, dim=-1)
    gt_loss = torch.full_like(loss_matrix_all_label_sum, fill_value=-1)

    for i in range(candidate_num):
        for j, label in enumerate(indication_y_s):
            gt_loss[i, j] = loss_matrix[i, j, int(label)]

    gt_loss_over_all_label = gt_loss / loss_matrix_all_label_sum

    # return [candidate_num, indication_num]
    return gt_loss_over_all_label
    pass


def main(logger, args):
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2)
    model = AutoModelForCausalLM.from_pretrained(args.gpt2)
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        model = model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    logger.info('ptm_name:{}'.format(args.gpt2))
    logger.info('ptm param_num: {}'.format(sum(p.numel() for p in model.parameters())))

    if args.train_task is None:
        # standard case where the training task and the test task are the same
        train_task = args.task
    else:
        raise NotImplementedError
        # zero-shot transfer case where the training task is different from the test task
        train_task = args.train_task
        assert args.do_check

    # datasets where the average input length is long
    long_datasets = ["cr", "subj", "agnews",
                     "amazon", "yelp_full", "yelp_binary", "boolq",
                     "dbpedia", "yahoo"]
    max_length = 256 if train_task in long_datasets else 128
    # batch_size = int(args.batch_size / 2) if train_task in long_datasets else args.batch_size
    batch_size = args.batch_size

    logger.info("%s %s" % (args.method, args.task))

    assert args.method in ["direct", "channel"]

    if args.use_demonstrations:
        assert args.do_zeroshot and not args.do_train

    if args.ensemble:
        assert args.use_demonstrations

    if args.do_train or args.use_demonstrations:
        assert args.train_seed > 0

    n_templates = 4
    k = int(args.k)
    seed = int(args.seed)

    # train_data = load_data(args.data_dir, train_task, k, seed, "train")
    # candidate_data = load_data_by_fp(args.task, args.candidate_fp)
    # indication_data = load_data_by_fp(args.task, args.indication_fp)
    train_data = load_data_by_fp(args.task)

    # for i, (x,y) in enumerate(train_data):
    #     if int(y) == 0:
    #         logger.info('the {} th data y == 0'.format(i))

    logger.info('label_set:{}'.format(Counter(map(lambda x:x[1],train_data))))

    # logger.info('{}\nfor debug, so use 200 of train data{}\n'.format('*'*80,'*'*80))
    # train_data = train_data[:200]

    train_data = list(enumerate(train_data))

    indication_data = copy.copy(train_data)

    tmp_candidate_num = len(train_data)

    indication_data_num_per_iteration = []
    candidate_data_num_per_iteration = []

    indication_num = args.initial_indication_set_size

    iter_num = 0

    while tmp_candidate_num > args.final_candidate_size:

        iter_num += 1
        indication_data_num_per_iteration.append(indication_num)
        candidate_data_num_per_iteration.append(tmp_candidate_num)
        if int(tmp_candidate_num / args.progressive_p) > args.final_candidate_size:
            indication_num = int(indication_num * args.progressive_p)
            tmp_candidate_num = int(tmp_candidate_num / args.progressive_p)
        else:
            break
    remaining_candidate_data_num_per_iteration = candidate_data_num_per_iteration[1:] + [args.final_candidate_size]
    # iter_num+=1
    # final_indication_num = int(len(train_data) * args.initial_indication_set_size / args.final_candidate_size )
    # indication_data_num_per_iteration.append(final_indication_num)
    # candidate_data_num_per_iteration.append(args.final_candidate_size)

    logger.info('iter_num:{}'.format(iter_num))
    for i in range(iter_num):
        tmp_candidate_num = candidate_data_num_per_iteration[i]
        logger.info(
            'at the {} th iter, the candidate_num = {}, the indication_num = {}, the total pair num = {},  remaining_candidate_num = {}'.
            format(i, tmp_candidate_num, indication_data_num_per_iteration[i],
                   tmp_candidate_num * indication_data_num_per_iteration[i],
                   remaining_candidate_data_num_per_iteration[i]))

    # random.seed(2)
    random.Random(args.indication_order_random_seed).shuffle(indication_data)

    now_candidate_data = train_data

    now_candidate_indication_pair_score_matrix = None

    tmp_counter = Counter()
    for idx, (x, y) in now_candidate_data:
        tmp_counter[y] += 1

    logger.info('whole training set label distribution:{}'.format(tmp_counter))

    n_classes = N_LABELS_DICT.get(args.task, None)
    output_dir = 'exps/tag_{}/{}/{}/{}/tempalte_{}/seed_{}_use_{}_p_{}_initial_i_size_{}_final_c_size_{}_balance_{}'.format(
        args.exp_tag,
        args.gpt2.replace('/','_'),
        args.task,
        args.method,
        args.template_idx,
        args.indication_order_random_seed,
        args.select_metric,
        args.progressive_p,
        args.initial_indication_set_size,
        args.final_candidate_size,
        args.select_example_label_balance,
    )
    if not args.select_most_useful:
        output_dir = output_dir + '_not_useful'
    os.makedirs(output_dir, exist_ok=True)
    all_indication_zero_shot_loss_matrix = None
    for i in range(iter_num):
        logger.info('start iter:{}'.format(i))
        now_cache_dir = '{}/iter_{}'.format(output_dir, i)
        os.makedirs(now_cache_dir, exist_ok=True)
        now_loss_cache_path = '{}/loss_matrix.pkl'.format(now_cache_dir)

        if i == 0:
            indication_s = 0
        else:
            indication_s = indication_data_num_per_iteration[i - 1]

        indication_e = indication_data_num_per_iteration[i]

        now_indication_data = indication_data[indication_s:indication_e]

        logger.info('indication num = {}'.format(len(now_indication_data)))

        now_indication_idx, now_indication_example = list(zip(*now_indication_data))

        logger.info('now_indication_idx:{}'.format(now_indication_idx))

        now_candidate_idx, now_candidate_example = list(zip(*now_candidate_data))

        now_candidate_before_filtering_cache_path = '{}/candidate_before_filtering.json'.format(now_cache_dir)

        json.dump(now_candidate_idx, open(now_candidate_before_filtering_cache_path, 'w'))

        now_indication_cache_path = '{}/indication.json'.format(now_cache_dir)

        json.dump(now_indication_idx, open(now_indication_cache_path, 'w'))

        now_loss_matrix = get_candidate_indication_pair_loss(
            args, args.task, tokenizer, model, now_candidate_example, now_indication_example,
            max_length, args.template_idx, now_loss_cache_path)

        # now_loss_matrix
        # [candidate_num+1, indication_num, n_classes]

        new_candidate_indication_pair_loss_matrix = now_loss_matrix[:-1]
        new_indication_zero_shot_loss_matrix = now_loss_matrix[-1]

        if all_indication_zero_shot_loss_matrix is None:
            all_indication_zero_shot_loss_matrix = new_indication_zero_shot_loss_matrix
        else:
            all_indication_zero_shot_loss_matrix = torch.cat(
                [all_indication_zero_shot_loss_matrix, new_indication_zero_shot_loss_matrix], dim=0)

        now_indication_zero_shot_loss_cache_path = '{}/indication_zero_shot_loss_matrix.pkl'.format(now_cache_dir)

        torch.save(all_indication_zero_shot_loss_matrix, open(now_indication_zero_shot_loss_cache_path, 'wb'))

        now_indication_x_s, now_indication_y_s = zip(*now_indication_example)

        if args.select_metric == 'loss':
            new_candidate_indication_score = transform_loss_matrix_into_final_loss(
                new_candidate_indication_pair_loss_matrix, now_indication_y_s, args, False)
        elif args.select_metric == 'acc':
            raise NotImplementedError
            new_candidate_indication_score = transform_loss_matrix_into_acc(
                new_candidate_indication_pair_loss_matrix, now_indication_y_s)

        assert new_candidate_indication_score.dim() == 2
        if now_candidate_indication_pair_score_matrix is None:
            now_candidate_indication_pair_score_matrix = new_candidate_indication_score
        else:
            now_candidate_indication_pair_score_matrix = \
                torch.cat([now_candidate_indication_pair_score_matrix,
                           new_candidate_indication_score], dim=1)

        if args.mask_same_candidate_indication_pair:

            now_candidate_idx_tensor = torch.tensor(now_candidate_idx, dtype=torch.long)
            now_all_indication_idx_tensor = torch.tensor(
                list(map(lambda x: x[0], indication_data[:indication_e]))
                , dtype=torch.long)

            now_candidate_idx_tensor = now_candidate_idx_tensor.unsqueeze(1)
            now_all_indication_idx_tensor = now_all_indication_idx_tensor.unsqueeze(0)

            candidate_indication_same_mask = (now_candidate_idx_tensor == now_all_indication_idx_tensor)

            assert candidate_indication_same_mask.size() == now_candidate_indication_pair_score_matrix.size(), \
                'now_candidate_indication_pair_score_matrix = {},\ncandidate_indication_same_mask = {}'. \
                    format(now_candidate_indication_pair_score_matrix.size(), candidate_indication_same_mask.size())

            now_candidate_indication_pair_score_matrix_masked = \
                torch.masked_fill(now_candidate_indication_pair_score_matrix, candidate_indication_same_mask, value=0)

            now_candidate_score_matrix_sum = torch.sum(now_candidate_indication_pair_score_matrix_masked, dim=1)

            now_candidate_score_matrix = now_candidate_score_matrix_sum / torch.sum(
                (~candidate_indication_same_mask).float(), dim=1)
            now_candidate_score_matrix = now_candidate_score_matrix.tolist()

        else:

            now_candidate_score_matrix = torch.mean(now_candidate_indication_pair_score_matrix, dim=1).tolist()

        now_candidate_score_matrix_and_idx = list(zip(now_candidate_score_matrix, now_candidate_idx))

        if args.select_metric == 'loss':
            if args.select_most_useful:
                now_candidate_score_matrix_and_idx.sort(key=lambda x: x[0])
            else:
                now_candidate_score_matrix_and_idx.sort(key=lambda x: x[0],reverse=True)
        elif args.select_metric == 'acc':
            logger.info('do not support acc')
            raise NotImplementedError
            now_candidate_score_matrix_and_idx.sort(key=lambda x: x[0], reverse=True)

        logger.info('now_candidate_score_matrix_and_idx:\n{}'.
                    format(now_candidate_score_matrix_and_idx[:20]))

        if args.select_example_label_balance:
            label_to_candidate_dict = {}
            for score, idx in now_candidate_score_matrix_and_idx:
                if train_data[idx][1][1] in label_to_candidate_dict:
                    label_to_candidate_dict[train_data[idx][1][1]].append(idx)
                else:
                    label_to_candidate_dict[train_data[idx][1][1]] = [idx]

            remaining_candidate_idx = []
            remaining_every_label_candidate_num = remaining_candidate_data_num_per_iteration[i] // n_classes
            for label, label_candidate in label_to_candidate_dict.items():
                remaining_candidate_idx.extend(label_candidate[:remaining_every_label_candidate_num])

            remaining_candidate_idx_set = set(remaining_candidate_idx)


        else:
            now_candidate_score_matrix_and_idx = now_candidate_score_matrix_and_idx[
                                                 :remaining_candidate_data_num_per_iteration[i]]
            remaining_candidate_idx = list(map(lambda x: x[1], now_candidate_score_matrix_and_idx))
            remaining_candidate_idx_set = set(remaining_candidate_idx)

        now_candidate_after_filtering_cache_path = \
            '{}/candidate_after_filtering.json'.format(now_cache_dir)

        json.dump(list(remaining_candidate_idx_set), open(now_candidate_after_filtering_cache_path, 'w'))

        # 为了看最终过滤得到的top分数的score，针对balance情况，非balance前面直接打印过了

        remaining_candidate_score_matrix_and_idx = list(filter(
            lambda x: x[1] in remaining_candidate_idx_set, now_candidate_score_matrix_and_idx
        ))

        logger.info('remaining_candidate_score_matrix_and_idx:\n{}'.
                    format(remaining_candidate_score_matrix_and_idx[:20]))

        # 过滤得到top分数的数据
        remaining_candidate_data = list(filter(
            lambda x: x[0] in remaining_candidate_idx_set, now_candidate_data
        ))

        # 统计过滤得到的类别分布
        tmp_counter = Counter()
        for idx, (x, y) in remaining_candidate_data:
            tmp_counter[y] += 1

        logger.info('remaining candidate label distribution:{}'.format(tmp_counter))

        # 同时保留top分数的数据的indication score
        now_candidate_indication_pair_score_matrix_idx = \
            list(zip(now_candidate_indication_pair_score_matrix, now_candidate_idx))

        assert len(now_candidate_indication_pair_score_matrix_idx) == len(now_candidate_indication_pair_score_matrix)

        remaining_candidate_indication_pair_score_matrix_idx = list(filter(
            lambda x: x[1] in remaining_candidate_idx_set, now_candidate_indication_pair_score_matrix_idx
        ))

        # logger.info()

        now_candidate_data = remaining_candidate_data
        now_candidate_indication_pair_score_matrix = list(
            map(lambda x: x[0].unsqueeze(0), remaining_candidate_indication_pair_score_matrix_idx))
        now_candidate_indication_pair_score_matrix = torch.cat(now_candidate_indication_pair_score_matrix, dim=0)

        logger.info('after iter:{}'.format(i, ))
        logger.info('now_candidate_data = {}'.format(len(now_candidate_data)))
        logger.info(
            'now_candidate_indication_pair_score_matrix = {}'.format(now_candidate_indication_pair_score_matrix.size()))

    now_cache_dir = '{}/final_result'.format(output_dir)
    os.makedirs(now_cache_dir, exist_ok=True)

    logger.info('start write final candidate result')
    final_candidate_indication_matrix_output_fp = '{}/final_candidate_indication_score_{}.pkl'.format(now_cache_dir,
                                                                                                      args.select_metric)
    torch.save(now_candidate_indication_pair_score_matrix, open(final_candidate_indication_matrix_output_fp, 'wb'))
    final_candidate_data_output_fp = '{}/final_candidate_data.json'.format(now_cache_dir)
    json.dump(now_candidate_data, open(final_candidate_data_output_fp, 'w'))
    logger.info('finish write final candidate result')

    all_indication_zero_shot_score_output_fp = '{}/final_indication_zero_shot_score_loss.pkl'.format(now_cache_dir)

    all_indication_zero_shot_score = \
        transform_loss_matrix_into_final_loss \
                (
                all_indication_zero_shot_loss_matrix.unsqueeze(0),
                list(map(lambda x: x[1][1], indication_data[:indication_data_num_per_iteration[-1]])),
                args,
                False
            )
    # all_indication_zero_shot_loss_matrix
    torch.save(all_indication_zero_shot_score, open(all_indication_zero_shot_score_output_fp, 'wb'))

    all_indication_idx = list(map(lambda x: x[0], indication_data[:indication_data_num_per_iteration[-1]]))
    all_indication_idx_output_fp = '{}/final_indication.json'.format(now_cache_dir)
    json.dump(all_indication_idx, open(all_indication_idx_output_fp, 'w'))

    # logger.info('tmp debug for iter running, so exit')

    # exit()

    # model = None
    # run(args, logger, args.do_train, args.do_zeroshot,
    #     args.task, train_task,
    #     k, seed, args.train_seed,
    #     args.out_dir, args.split,
    #     tokenizer, model, candidate_data, indication_data,
    #     batch_size, max_length, args.gpt2,
    #     args.template_idx, args.method,
    #     args.lr, args.warmup_steps,
    #     use_demonstrations=args.use_demonstrations,
    #     use_calibration=args.use_calibration,
    #     ensemble=args.ensemble,
    #     is_null=args.split is None,
    #     prompt_tune=args.prompt_tune,
    #     head_tune=args.head_tune,
    #     transform_tune=args.transform_tune,
    #     do_check=args.do_check,
    #     n_prefix=args.n_prefix)


def evaluate(dev_data, label_losses):
    labels = list(label_losses.keys())
    acc = []
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l: np.sum(label_losses[l][idx]) if isinstance(label_losses[l], np.ndarray) else torch.sum(
            label_losses[l][idx]) for l in label_losses}
        prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        acc.append(prediction == label)
    return np.mean(acc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument()

    # parser.add_argument('--')
    # parser.add_argument('--select_label_balance',required=True)
    parser.add_argument('--select_most_useful',type=int,default=1)
    parser.add_argument('--mask_same_candidate_indication_pair', type=int, required=True)
    parser.add_argument('--exp_tag', required=True)
    parser.add_argument('--select_example_label_balance', type=int, required=True)
    parser.add_argument('--select_metric', required=True, choices=['acc', 'loss'])
    parser.add_argument('--indication_order_random_seed', type=int, required=True)
    parser.add_argument('--progressive_p', type=float, required=True)
    parser.add_argument('--initial_indication_set_size', type=int, required=True)
    parser.add_argument('--final_candidate_size', type=int, required=True)
    # parser.add_argument('--indicate_example_from',choices=['filtered_candidate','random_sample_from_whole_dataset'])
    # parser.add_argument('--indicate_example_from',choices=['filtered','random'])
    # filtered是从被过滤掉的不重要的样本里选择indication
    # randon就是直接从全局选择不重要的样本，缺点是被选择用来当 indication 的样本就不会被用来当candidate了（因为p(y1|x1,y1,x1)肯定偏高）
    # 好像random也不会造成无法被选为candidate，只要pair中，两个样本相同的loss或者p不计入平均就完事了。
    parser.add_argument('--indicate_example_from', choices=['random'], required=True)

    # parser.add_argument('--use_fitlog_record_indication_performance',type=int,required=True)
    # parser.add_argument('--most_useful_sample_proportion',type=float,required=True)
    parser.add_argument('--debug_mode_trial_num', type=int, default=5)
    parser.add_argument('--compare_debug_mode', type=int, default=0)
    # parser.add_argument('--select_most_useful',required=True,type=str,choices=['0','1','random']) #如果1，那么选能够使indication做的最对的candidate
    # parser.add_argument("--candidate_fp", type=str, required=True)
    # parser.add_argument('--indication_fp',type=str, required=True)
    # parser.add_argument('--select_example_num_every_label',type=int,required=True)
    parser.add_argument("--method", type=str, required=True, choices=['channel', 'direct'])
    parser.add_argument('--template_idx', required=True, type=int)

    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--do_check", default=False, action="store_true")

    # parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument('--use_calibration', type=int, default=0)
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--ensemble", default=False, action="store_true")
    parser.add_argument("--prompt_tune", default=False, action="store_true")
    parser.add_argument("--head_tune", default=False, action="store_true")
    parser.add_argument("--transform_tune", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--train_task", type=str, default=None)

    parser.add_argument("--k", type=str, default="16")
    parser.add_argument("--seed", type=str, default="100")
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--n_prefix", type=int, default=20)
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    args = parser.parse_args()


    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    # logger.info(args)

    # 打印args，基础操作
    logger.info('start print args')
    for k, v in args.__dict__.items():
        logger.info('{}:{}'.format(k, v))

    # if not args.use_fitlog_record_indication_performance:
    fitlog.debug()

    fitlog.add_hyper(args)

    main(logger, args)
    fitlog.finish()
