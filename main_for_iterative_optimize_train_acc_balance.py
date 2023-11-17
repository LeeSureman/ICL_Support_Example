import argparse
import os

import fitlog
import torch
import logging

handlers = [logging.StreamHandler()]
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=handlers)

logger = logging.getLogger(__name__)

import random
import json
import copy
import numpy as np
from main_for_progressive_examples_selection import load_data_by_fp,N_LABELS_DICT
from my_dataset import MyDataset
from run import my_inference
from util import get_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from main_for_iterative_optimize_heuristic_score_func import iterative_optimize_heuristic_score_function
import tqdm
from collections import Counter
from data import prepare_data
from run import inference
import fitlog
import pickle
# from fastNLP import cache_result

os.makedirs('fitlog_search_examples', exist_ok=True)
fitlog.set_log_dir('fitlog_search_examples')




def count_example_idxs_label(example_idxs, candidate_y):
    counter = Counter()
    for idx in example_idxs:
        counter[candidate_y[idx]] += 1
    return counter


def normalize_std(inp_tensor):
    mean_value = torch.mean(inp_tensor)
    inp_tensor_subtract_mean = inp_tensor - mean_value
    std_value = torch.std(inp_tensor_subtract_mean)

    output_tensor = inp_tensor_subtract_mean / std_value
    return output_tensor


def normalize_01(inp_tensor):
    max_value = torch.max(inp_tensor)
    min_value = torch.min(inp_tensor)

    output_tensor = (inp_tensor - min_value) / (max_value - min_value)
    return output_tensor
    pass


def calculate_example_idxs_score(whole_feature_mean, whole_feature_sim, example_idxs, args):
    # logger.info('in example_idxs_score, example_idxs:{}'.format(example_idxs))
    tmp_feature_mean = whole_feature_mean[example_idxs]
    tmp_feature_sim = whole_feature_sim[example_idxs][:, example_idxs]

    tmp_feature_mean_score = torch.mean(tmp_feature_mean)
    tmp_diversity_score = - torch.mean(tmp_feature_sim)
    tmp_feature_score = tmp_feature_mean_score + tmp_diversity_score * args.diversity_score_scale

    return {'total': tmp_feature_score, 'importance': tmp_feature_mean_score, 'diversity': tmp_diversity_score}


def evaluate(dev_data, label_losses):
    labels = list(label_losses.keys())
    acc = []
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l: np.sum(label_losses[l][idx]) if isinstance(label_losses[l], np.ndarray) else torch.sum(
            label_losses[l][idx]) for l in label_losses}
        prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        acc.append(prediction == label)
    return np.mean(acc)


# @cache
def verify_example_idxs_list_on_indication_data(all_candidate_data, example_idxs_list, indication_examples, args,
                                                tokenizer, model, now_iter=None):
    logger.info('indication_examples:{}'.format(len(indication_examples)))
    template = get_prompts(args.task, args.template_idx)

    all_candidate_examples = list(map(lambda x: x[1], all_candidate_data))

    candidate_examples_list = []

    for example_idxs in example_idxs_list:
        candidate_examples = []
        for idx in example_idxs:
            candidate_examples.append(all_candidate_examples[idx])

        candidate_examples_list.append(candidate_examples)

    mydataset = MyDataset(args, candidate_examples_list, indication_examples,
                          args.method, tokenizer,
                          max_length_per_example=args.max_length_per_example,
                          n_classes=N_LABELS_DICT[args.task], template=template,
                          add_zero_shot_pseudo_candidate=1,
                          candidate_demonstrations_element_check_tag='list',
                          max_length=args.max_length)

    # now_iter_loss_cache_path = '{}/iter_{}'.format(args.output_dir, now_iter, )

    example_idxs_list_loss_matrix = my_inference(model, mydataset, args.final_batch_size)
    logger.info('example_idxs_list_loss_matrix:{}'.format(example_idxs_list_loss_matrix.size()))
    # [len(example_idxs_list)+1,indication_num,n_classes]

    example_idxs_list_acc = []
    candidate_num = len(example_idxs_list)
    for i in range(candidate_num + 1):
        tmp_example_idxs_loss_matrix = example_idxs_list_loss_matrix[i]
        tmp_example_idxs_loss_matrix = torch.transpose(tmp_example_idxs_loss_matrix, 0, 1)

        acc = evaluate(indication_examples, {str(i): loss for i, loss in enumerate(tmp_example_idxs_loss_matrix)})
        example_idxs_list_acc.append(acc)

    result = {}
    result['acc'] = example_idxs_list_acc
    result['example_idxs_list'] = example_idxs_list
    result['indication_examples'] = indication_examples
    result['candidate_examples_list'] = candidate_examples_list

    result['loss_matrix'] = example_idxs_list_loss_matrix

    if args.method == 'direct':
        bias_losses = []
        n_classes = N_LABELS_DICT[args.task]
        for candidate_examples in candidate_examples_list + [[]]:
            bias_input_tensors = prepare_data(
                tokenizer, candidate_examples, None,
                max_length=args.max_length,
                max_length_per_example=args.max_length_per_example,
                n_classes=n_classes,
                templates=get_prompts(args.task, args.template_idx),
                method_type=args.method,
                use_demonstrations=True,
                ensemble=False,
                is_null=True,
                shuffle_seed=-1)

            for input_tensor in bias_input_tensors:
                bias_losses.append(inference(model,
                                             input_tensor,
                                             batch_size, use_tqdm=False)[0])

        bias_losses = torch.tensor(bias_losses).view(candidate_num + 1, n_classes)
        result['bias_loss_matrix'] = bias_losses
        example_idxs_list_acc_calibration = []
        for i in range(candidate_num + 1):
            tmp_example_idxs_loss_matrix = example_idxs_list_loss_matrix[i]
            # tmp_example_idxs_loss_matrix [indication_num, n_classes]
            bias_loss = bias_losses[i].unsqueeze(0)
            tmp_example_idxs_loss_matrix = tmp_example_idxs_loss_matrix - bias_loss
            tmp_example_idxs_loss_matrix = torch.transpose(tmp_example_idxs_loss_matrix, 0, 1)

            acc_calibration = evaluate(indication_examples,
                                       {str(i): loss for i, loss in enumerate(tmp_example_idxs_loss_matrix)})
            example_idxs_list_acc_calibration.append(acc_calibration)

        result['acc_calibration'] = example_idxs_list_acc_calibration
        # loss_matrix_calibration = example_idxs_list_loss_matrix[:-1].numpy() - bias_losses

        # tmp_example_idxs_loss_matrix_calibration =

    return result
    # [candidate_example_seq_num,indication_num,n_classes]
    # candidate_example_seq_num就是待评测的candidate_example_seq数量，一个candidate_example_seq_num包含了k个example


class Optimize_Heuristic_Score_Config:
    def __init__(self, args):
        self.num_iteration = 20
        self.beam_size = 16
        self.diversity_score_scale = args.diversity_score_scale
        self.label_balance = args.label_balance
        self.candidate_example_num_every_label = args.candidate_example_num_every_label
        self.candidate_example_num_total = args.candidate_example_num_total
        self.sample_topk_range = args.sample_topk_range
        self.task = args.task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_at_which_iter', choices=['every_iter', 'last', 'first_last'], default='first_last')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--ptm_name', choices=['gpt2-large', 'gpt2-medium','EleutherAI/gpt-neo-1.3B'], required=True)

    parser.add_argument('--direct_plus', type=int, required=True)

    parser.add_argument('--num_iteration', type=int, required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--which_normalization', choices=['std', '01', 'no'], required=True)
    parser.add_argument('--diversity_score_scale', type=float, required=True)
    parser.add_argument('--label_balance', type=int, required=True)
    parser.add_argument('--candidate_example_num_every_label', type=int, default=-1)
    parser.add_argument('--candidate_example_num_total', type=int, default=-1)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--which_diversity_feature', required=True, choices=['sbert_embedding', 'lm_feedback'])
    parser.add_argument('--beam_size', type=int, required=True)
    parser.add_argument('--shuffle_order_num_in_beam', type=int, required=True)
    parser.add_argument('--select_metric', choices=['p', 'loss'], required=True)
    parser.add_argument('--sample_topk_range', type=int, required=True)
    parser.add_argument('--num_indication', type=int, required=True)
    parser.add_argument('--train_data_shuffle_and_select_indication_data_seed', type=int, required=True)
    parser.add_argument('--verify_metric', choices=['acc', 'loss'], required=True)
    parser.add_argument('--debug_candidate_size', type=int, default=-1)
    parser.add_argument('--initial_example_idxs', choices=['random', 'searched_from_heuristic_score'], required=True)
    parser.add_argument('--initial_example_idxs_repeat_limit', type=int, required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--cache_hash', default='-1')
    args = parser.parse_args()
    args.num_indication = 100

    if args.output_dir is None:
        args.output_dir = '{}/search_dir'.format(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    args.task = args.input_dir.split('/')[3]

    # if 'gpt'

    task_test_fp = 'test_data/{}/test.tsv'.format(args.task)

    if not os.path.exists(task_test_fp):
        task_test_fp = 'test_data/{}/test.csv'.format(
            args.task)

    test_examples = load_data_by_fp(args.task, task_test_fp)

    args.template_idx = int(args.input_dir.split('/')[5].replace('template_', '').replace('tempalte_', ''))
    assert args.template_idx >= 0 and args.template_idx <= 3, args.template_idx

    assert args.ptm_name.replace('/','_') == args.input_dir.split('/')[2], \
        'args.ptm_name={}, ptm_name in input_dir={}'.format(args.ptm_name, args.input_dir.split('/')[2])
    args.method = args.input_dir.split('/')[4]

    assert args.verify_metric == 'acc'

    long_datasets = ["cr", "subj", "agnews",
                     "amazon", "yelp_full", "yelp_binary", "boolq",
                     "dbpedia", "yahoo"]
    max_length_per_example = 256 if args.task in long_datasets else 128
    args.max_length_per_example = max_length_per_example
    batch_size = int(args.batch_size / 2) if args.task in long_datasets else args.batch_size

    mem = args.batch_size * max_length_per_example
    n_classes = N_LABELS_DICT[args.task]

    if args.label_balance:
        k = args.candidate_example_num_every_label * N_LABELS_DICT[args.task]
        args.candidate_example_num_total = k
    else:
        k = args.candidate_example_num_total
        args.candidate_example_num_every_label = k // N_LABELS_DICT[args.task]


    if n_classes == 2:
        max_length = max_length_per_example * (k+1)
    elif n_classes in [4, 5]:
        max_length = int(max_length_per_example * 1.5 * (k+1))
    elif n_classes in [6]:
        max_length = int(max_length_per_example * 2 * (k+1))
    else:
        max_length = 1024

    max_length = min(max_length, 1024)
    batch_size = int(mem / max_length)

    args.final_batch_size = batch_size
    args.max_length = max_length

    logger.info('now print args')
    logger.info(args)

    logger.info('total calculate count:{}'.format(args.beam_size * args.beam_size * args.num_iteration))

    fitlog.set_rng_seed(args.seed)

    final_candidate_indication_score = torch.load(open(
        '{}/final_result/final_candidate_indication_score_loss.pkl'.format(args.input_dir), 'rb'
    ), map_location='cpu')

    final_indication_zero_shot_score = torch.load(open(
        '{}/final_result/final_indication_zero_shot_score_loss.pkl'.format(args.input_dir), 'rb'
    ), map_location='cpu')
    # fitlog.debug()

    # if args.cache_hash == '':
    #     args.cache_hash = hash(args)
    tmp_args_dict = args.__dict__
    tmp_args_dict['test_at_which_iter'] = 'first_last'

    args_k_v_list = sorted(list(tmp_args_dict.items()))
    import hashlib
    md5_algorithm = hashlib.md5()

    args_str = '_'.join(list(map(lambda x:str(x[0])+str(x[1]),args_k_v_list)))
    md5_algorithm.update(args_str.encode('utf-8'))
    args_hash = md5_algorithm.hexdigest()
    # args_k_v_tuple = tuple(args_k_v_list)
    # args_hash = hash(args_k_v_tuple)

    if args.cache_hash == '-1':
        args.cache_hash = args_hash

    logger.info('hash:{}'.format(args_hash))

    args.output_dir = '{}/{}'.format(args.output_dir,args.cache_hash)
    os.makedirs(args.output_dir,exist_ok=True)

    with open('{}/args.json'.format(args.output_dir),'w') as f:
        json.dump(args_k_v_list,f)

    fitlog.add_hyper(args)

    candidate_data = json.load(open('{}/final_result/final_candidate_data.json'.format(args.input_dir), 'r'))

    # if args.debug_candidate_size > 0:
    #     logger.warning(
    #         'args.debug_candidate_size={}, so only search in partial candidate data'.format(args.debug_candidate_size))
    #     logger.warning(
    #         'args.debug_candidate_size={}, so only search in partial candidate data'.format(args.debug_candidate_size))
    #     logger.warning(
    #         'args.debug_candidate_size={}, so only search in partial candidate data'.format(args.debug_candidate_size))
    #     logger.info(
    #         'args.debug_candidate_size={}, so only search in partial candidate data'.format(args.debug_candidate_size))
    #     logger.info(
    #         'args.debug_candidate_size={}, so only search in partial candidate data'.format(args.debug_candidate_size))
    #     logger.info(
    #         'args.debug_candidate_size={}, so only search in partial candidate data'.format(args.debug_candidate_size))

    candidate_original_idx, candidate_example = list(zip(*candidate_data))

    # 加载一下train集，从中选取这次搜索的验证集

    train_data = load_data_by_fp(args.task)

    train_data_with_original_dx = list(enumerate(train_data))

    candidate_original_idx_set = set(candidate_original_idx)

    logger.info('train_data_with_original_dx:{}'.format(len(train_data_with_original_dx)))

    train_data_with_original_dx_filtered = list(
        filter(lambda x: x[0] not in candidate_original_idx_set, train_data_with_original_dx))

    logger.info('train_data_with_original_dx_filtered:{}'.format(len(train_data_with_original_dx_filtered)))

    random.Random(args.train_data_shuffle_and_select_indication_data_seed).shuffle(train_data_with_original_dx_filtered)
    indication_data_from_train = train_data_with_original_dx_filtered[:args.num_indication]

    pickle.dump(indication_data_from_train, open('{}/indication_data_from_train.pkl'.format(args.output_dir),'wb'))

    logger.info('indication_data_from_train:')
    for i in range(5):
        logger.info('indication_data_from_train[{}]: {}'.format(i, indication_data_from_train[i]))

    indication_examples_from_train = list(map(lambda x: x[1], indication_data_from_train))

    indication_examples_idx_from_train = list(map(lambda x: x[0], indication_data_from_train))

    logger.info('indication_examples_idx_from_train:{}'.format(indication_examples_idx_from_train))

    candidate_x, candidate_y = list(zip(*candidate_example))

    candidate_data_with_new_idx = list(enumerate(candidate_example))

    label_to_candidate_idx = {}

    for i, (x, y) in candidate_data_with_new_idx:
        if y not in label_to_candidate_idx:
            label_to_candidate_idx[y] = [i]
        else:
            label_to_candidate_idx[y].append(i)

    if args.which_diversity_feature == 'lm_feedback':
        if args.select_metric == 'loss':
            feature = final_indication_zero_shot_score - final_candidate_indication_score
        elif args.select_metric == 'p':
            feature = final_candidate_indication_score - final_indication_zero_shot_score
        else:
            raise NotImplementedError
    elif args.which_diversity_feature == 'sbert_embedding':
        raise NotImplementedError
    else:
        raise NotImplementedError

    feature_mean = torch.mean(feature, dim=1)

    # feature_mean [candidate_num]

    feature_after_norm = torch.nn.functional.normalize(feature, p=2, dim=1)
    feature_sim = torch.matmul(feature_after_norm, feature_after_norm.T)

    if args.which_normalization == '01':
        feature_mean = normalize_01(feature_mean)
        feature_sim = normalize_01(feature_sim)
        pass
    elif args.which_normalization == 'std':
        feature_mean = normalize_std(feature_mean)
        feature_sim = normalize_std(feature_sim)
        pass
    elif args.which_normalization == 'no':
        pass
    else:
        raise NotImplementedError

    candidate_num = feature.size(0)
    # logger.info('tmp demo so initialize examples by random')

    initial_example_idxs_list = []
    if args.initial_example_idxs == 'random':
        for i in range(args.beam_size):

            if args.label_balance:
                n_classes = N_LABELS_DICT[args.task]
                tmp_example_idxs = []
                for i in range(n_classes):
                    tmp_example_idxs.extend(
                        random.sample(label_to_candidate_idx[str(i)], args.candidate_example_num_every_label)
                    )

                initial_example_idxs_list.append(tmp_example_idxs)

            else:
                initial_example_idxs_list.append(
                    random.sample(list(range(candidate_num)), args.candidate_example_num_total))
    elif args.initial_example_idxs == 'searched_from_heuristic_score':

        optimize_heuristic_score_config = Optimize_Heuristic_Score_Config(args)

        new_iter_example_idxs_seqscore, new_iter_example_idxs_list, history_example_idxs_and_score_dict = iterative_optimize_heuristic_score_function(
            optimize_heuristic_score_config, feature_mean, feature_sim, initial_size_scale=10,
            label_to_candidate_idx=label_to_candidate_idx, candidate_y=candidate_y)

        used_example_idxs_set = set()

        logger.info('history_example_idxs_and_score_dict size: {}'.format(len(history_example_idxs_and_score_dict)))
        history_example_idxs_and_score_list = list(history_example_idxs_and_score_dict.items())

        history_example_idxs_and_score_list.sort(key=lambda x: x[1], reverse=True)

        logger.info('*****start looking for unique example_idxs candidate by topk heuristic score*****')
        found_example_idxs_num = 0
        initial_example_idxs_score_list = []
        for example_idxs, example_idxs_score in tqdm.tqdm(history_example_idxs_and_score_list):
            if len(set(example_idxs) & used_example_idxs_set) <= args.initial_example_idxs_repeat_limit:
                initial_example_idxs_list.append(list(example_idxs))
                initial_example_idxs_score_list.append(example_idxs_score)

                found_example_idxs_num += 1

                used_example_idxs_set = used_example_idxs_set.union(set(example_idxs))

            if found_example_idxs_num >= args.beam_size:
                break

        for i in range(len(initial_example_idxs_list)):
            random.shuffle(initial_example_idxs_list[i])

        logger.info('initialize example idxs by heuristic score.')
        for i, (tmp_score, tmp_example_idxs) in enumerate(
                list(zip(new_iter_example_idxs_seqscore, new_iter_example_idxs_list))):
            logger.info('top {} initialized, score: {} example_idxs: {}'.format(i, tmp_score, sorted(tmp_example_idxs)))

        logger.info('find {} example_idxs repeat <= {}'.format(len(initial_example_idxs_list),
                                                               args.initial_example_idxs_repeat_limit))

        for i in range(len(initial_example_idxs_list)):
            logger.info(
                'the {} th example_idxs score : {} ; example_idxs:{}'.format(i, initial_example_idxs_score_list[i],
                                                                             initial_example_idxs_list[i]))
        for i in range(args.beam_size - len(initial_example_idxs_list)):
            if args.label_balance:
                n_classes = N_LABELS_DICT[args.task]
                tmp_example_idxs = []
                for i in range(n_classes):
                    tmp_example_idxs.extend(
                        random.sample(label_to_candidate_idx[str(i)], args.candidate_example_num_every_label)
                    )

                initial_example_idxs_list.append(tmp_example_idxs)

            else:
                initial_example_idxs_list.append(
                    random.sample(list(range(candidate_num)), args.candidate_example_num_total))

        logger.info('only use top {} (beam_size) for initialization'.format(args.beam_size))

        logger.info('by heuristic score, using these example_idxs as initialization')

        # logger.info('new_iter_example_idxs_seqscore')

    else:
        raise NotImplementedError

    total_calculate_count = 0
    now_example_idxs_list = initial_example_idxs_list

    for i, example_idxs in enumerate(initial_example_idxs_list):
        logger.info('initial_example_idxs_list[{}] label counter: {}'.format(i, count_example_idxs_label(example_idxs,
                                                                                                         candidate_y)))

    logger.info('initial_example_idxs_list: {}'.format(len(initial_example_idxs_list)))

    model = AutoModelForCausalLM.from_pretrained(args.ptm_name)
    tokenizer = AutoTokenizer.from_pretrained(args.ptm_name)

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    history_example_idxs_set = set()

    tmp_zero_shot_result = verify_example_idxs_list_on_indication_data(candidate_data,
                                                                       [],
                                                                       indication_examples_from_train,
                                                                       args,
                                                                       tokenizer,
                                                                       model
                                                                       )
    tmp_zero_shot_acc = tmp_zero_shot_result['acc']
    logger.info('zero_shot acc:{}'.format(tmp_zero_shot_acc))
    if args.method == 'direct':
        tmp_zero_shot_acc_calibration = tmp_zero_shot_result['acc_calibration']
        logger.info('zero_shot acc calibration:{}'.format(tmp_zero_shot_acc_calibration))

    # exit()

    for i in range(args.num_iteration):
        new_iter_example_idxs_list = []
        for j, example_idxs in enumerate(now_example_idxs_list):
            example_idxs_set = set(example_idxs)
            for k in range(args.beam_size - args.shuffle_order_num_in_beam):
                tmp_replace_pos = random.choice(list(range(len(example_idxs))))
                remaining_example_idxs = copy.deepcopy(example_idxs)
                del remaining_example_idxs[tmp_replace_pos]

                # 开始计算剩下的candidate的分数
                feature_sim_on_remaining_example = feature_sim[:, remaining_example_idxs]
                # logger.info('feature_sim_on_remaining_example:{}'.format(feature_sim_on_remaining_example.size()))
                feature_diversity_score_on_remaining_example = - torch.mean(feature_sim_on_remaining_example, dim=1)
                feature_score = feature_mean + \
                                feature_diversity_score_on_remaining_example * args.diversity_score_scale

                # logger.info('feature_score:{}'.format(feature_score.size()))

                feature_score_with_idx = list(enumerate(feature_score.tolist()))
                feature_score_with_idx = list(filter(lambda x: x[0] not in example_idxs_set,
                                                     feature_score_with_idx))

                if args.label_balance:
                    feature_score_with_idx = list(filter(
                        lambda x: candidate_y[x[0]] == candidate_y[example_idxs[tmp_replace_pos]],
                        feature_score_with_idx, ))

                feature_score_with_idx.sort(key=lambda x: x[1], reverse=True)
                feature_score_with_idx = feature_score_with_idx[:args.sample_topk_range]

                tmp_feature_score = list(map(lambda x: x[1], feature_score_with_idx))
                tmp_idx = list(map(lambda x: x[0], feature_score_with_idx))

                tmp_p = torch.nn.functional.softmax(torch.tensor(tmp_feature_score), dim=0)
                # logger.info('tmp_p:{}'.format(tmp_p))
                # logger.info('tmp_p:{}'.format(tmp_p.size()))
                # tmp_p[-1] = 1 - torch.sum(tmp_p[:-1])
                tmp_p = tmp_p.numpy()
                tmp_p /= tmp_p.sum()
                # logger.info('tmp_p:{}'.format(tmp_p))
                # logger.info('tmp_p sum:{}'.format(np.sum(tmp_p)))

                choosed_example_idx = np.random.choice(tmp_idx, p=tmp_p)

                # compare random
                # choosed_example_idx = np.random.choice(list(filter(lambda x:x not in example_idxs_set, list(range(candidate_num)))))

                result_example_idxs = copy.deepcopy(example_idxs)
                result_example_idxs[tmp_replace_pos] = choosed_example_idx

                new_iter_example_idxs_list.append(result_example_idxs)
            for k in range(args.shuffle_order_num_in_beam):
                result_example_idxs = copy.deepcopy(example_idxs)
                random.shuffle(result_example_idxs)
                new_iter_example_idxs_list.append(result_example_idxs)

        logger.info('in iter {} new_example_idxs_seq_num:{} now_example_idxs_list: {} total: {}'.format \
                        (i,
                         len(new_iter_example_idxs_list),
                         len(now_example_idxs_list),
                         len(new_iter_example_idxs_list) + len(now_example_idxs_list))
                    )

        new_iter_example_idxs_list.extend(now_example_idxs_list)

        new_iter_example_idxs_list = list(map(lambda x: tuple(x), new_iter_example_idxs_list))
        # new_iter_example_idxs_list = list(filter(lambda x:x not in history_example_idxs_set, new_iter_example_idxs_list) )
        # history_example_idxs_set = history_example_idxs_set.union(set(new_iter_example_idxs_list))
        new_iter_example_idxs_list = list(set(new_iter_example_idxs_list))
        new_iter_example_idxs_list = list(map(lambda x: list(x), new_iter_example_idxs_list))

        # new_iter_example_idxs_list_with_seqidx = list(enumerate(new_iter_example_idxs_list))

        for j, example_idxs in enumerate(new_iter_example_idxs_list[:5]):
            logger.info('iter {}, new_iter_example_idxs_list[{}] label counter: {}'.format(i, j,
                                                                                           count_example_idxs_label(
                                                                                               example_idxs,
                                                                                               candidate_y)))

        logger.info('in iter {} total_after deduplication: {}'.format(i, len(new_iter_example_idxs_list)))

        total_calculate_count += len(new_iter_example_idxs_list)

        # new_iter_example_idxs_seqscore = []
        # now_iter_example_idx_seqscore_importance = []
        # now_iter_example_idx_seqscore_diversity = []
        # for j, example_idxs in new_iter_example_idxs_list_with_seqidx:
        #     tmp_score_dict = example_idxs_score(feature_mean, feature_sim, example_idxs, args)
        #
        #     new_iter_example_idxs_seqscore.append(tmp_score_dict['total'])
        #     now_iter_example_idx_seqscore_importance.append(tmp_score_dict['importance'])
        #     now_iter_example_idx_seqscore_diversity.append(tmp_score_dict['diversity'])
        os.makedirs('{}/iter_{}'.format(args.output_dir, i), exist_ok=True)
        example_idxs_list_result_cache_fp = '{}/iter_{}/example_idxs_list_result.pkl'.format(args.output_dir, i)
        if os.path.exists(example_idxs_list_result_cache_fp):
            example_idxs_list_result = pickle.load(open(example_idxs_list_result_cache_fp, 'rb'))
        else:
            example_idxs_list_result = verify_example_idxs_list_on_indication_data(candidate_data,
                                                                                   new_iter_example_idxs_list,
                                                                                   indication_examples_from_train,
                                                                                   args,
                                                                                   tokenizer,
                                                                                   model
                                                                                   )
            pickle.dump(example_idxs_list_result, open(example_idxs_list_result_cache_fp, 'wb'))
        if args.direct_plus:
            example_idxs_list_acc = example_idxs_list_result['acc_calibration']
        else:
            example_idxs_list_acc = example_idxs_list_result['acc']
        # example_idxs_list_acc_calibration = example_idxs_list_result['acc_calibration']
        if i == 0:
            logger.info('zero_shot_acc:{}'.format(example_idxs_list_acc[-1]))
            fitlog.add_best_metric({'zero_shot_acc': 100*example_idxs_list_acc[-1]})

            if args.method == 'direct':
                example_idxs_list_acc_calibration = example_idxs_list_result['acc_calibration']
                logger.info('zero_shot_acc_calibration:{}'.format(example_idxs_list_acc_calibration[-1]))
                fitlog.add_best_metric({'zero_shot_acc_calibration': 100*example_idxs_list_acc_calibration[-1]})

        example_idxs_list_acc = example_idxs_list_acc[:-1]

        example_idxs_list_acc = torch.tensor(example_idxs_list_acc)
        example_idxs_topk = torch.topk(example_idxs_list_acc, k=args.beam_size)

        example_idxs_topk_indices = example_idxs_topk.indices.tolist()
        example_idxs_topk_acc = example_idxs_topk.values.tolist()

        logger.info('iteration_{} example_idxs_topk_acc:{}'.format(i, example_idxs_topk_acc))
        for j, v in enumerate(example_idxs_topk_acc):
            fitlog.add_best_metric({'acc': {'top_{}'.format(j + 1): 100*v}}, name='iter_{}'.format(i))

        if args.method == 'direct':
            example_idxs_list_acc_calibration = example_idxs_list_result['acc_calibration']
            example_idxs_topk_acc_calibration = []
            for j in range(args.beam_size):
                example_idxs_topk_acc_calibration.append(
                    example_idxs_list_acc_calibration[example_idxs_topk_indices[j]])

            logger.info(
                'iteration_{} example_idxs_topk_acc_calibration:{}'.format(i, example_idxs_topk_acc_calibration))
            for j, v in enumerate(example_idxs_topk_acc_calibration):
                fitlog.add_best_metric({'acc_calib': {'top_{}'.format(j + 1): 100*v}}, name='iter_{}'.format(i))
        now_example_idxs_list = []

        for indice in example_idxs_topk_indices:
            now_example_idxs_list.append(new_iter_example_idxs_list[indice])
        if args.test_at_which_iter == 'every' or \
                (args.test_at_which_iter == 'first' and i == 0) or \
                (args.test_at_which_iter == 'last' and i == args.num_iteration - 1) or \
                (args.test_at_which_iter == 'first_last' and i in [0, args.num_iteration - 1]):

            example_idxs_list_test_result_cache_fp = '{}/iter_{}/example_idxs_list_test_result.pkl'.format(args.output_dir, i)
            if os.path.exists(example_idxs_list_test_result_cache_fp):
                example_idxs_list_test_result = pickle.load(open(example_idxs_list_test_result_cache_fp,'rb'))
            else:
                example_idxs_list_test_result = verify_example_idxs_list_on_indication_data(candidate_data,
                                                                                            now_example_idxs_list,
                                                                                            test_examples,
                                                                                            args,
                                                                                            tokenizer,
                                                                                            model
                                                                                            )
                pickle.dump(example_idxs_list_test_result, open(example_idxs_list_test_result_cache_fp,'wb'))
            if args.direct_plus:
                example_idxs_list_test_acc = example_idxs_list_test_result['acc_calibration']
            else:
                example_idxs_list_test_acc = example_idxs_list_test_result['acc']
            # logger.info('zero_shot_test_acc:{}'.format(example_idxs_list_test_acc[-1]))
            if i == 0:
                logger.info('iteration_{} zero_shot_test_acc:{}'.format(i, example_idxs_list_test_acc[-1]))
                fitlog.add_best_metric({'zero_shot_test_acc': 100*example_idxs_list_test_acc[-1]})

            logger.info('iteration_{} example_idxs_topk_test_acc:{}'.format(i, example_idxs_list_test_acc[:-1]))

            tmp_acc_list = []
            for j, v in enumerate(example_idxs_list_test_acc[:-1]):
                tmp_acc_list.append(v)
                fitlog.add_best_metric({'test_acc': {'top_{}'.format(j + 1): 100*v}}, name='iter_{}'.format(i))
                fitlog.add_best_metric({'test_acc_mean': {'top_{}'.format(j + 1): 100*sum(tmp_acc_list)/len(tmp_acc_list)}}, name='iter_{}'.format(i))

            if args.method == 'direct':
                example_idxs_list_test_acc_calibration = example_idxs_list_test_result['acc_calibration']
                logger.info('zero_shot_test_acc_calibration:{}'.format(example_idxs_list_test_acc_calibration[-1]))
                fitlog.add_best_metric({'zero_shot_test_acc_calibration': 100*example_idxs_list_test_acc_calibration[-1]})

                if i == 0:
                    logger.info('iteration_{} zero_shot_test_acc_calibration:{}'.format(i,
                                                                                        example_idxs_list_test_acc_calibration[
                                                                                            -1]))
                logger.info('iteration_{} example_idxs_topk_test_acc_calibration:{}'.format(i,
                                                                                            example_idxs_list_test_acc_calibration[
                                                                                            :-1]))
                tmp_acc_list = []
                for j, v in enumerate(example_idxs_list_test_acc_calibration[:-1]):
                    tmp_acc_list.append(v)
                    fitlog.add_best_metric({'test_acc_calib': {'top_{}'.format(j + 1): 100*v}}, name='iter_{}'.format(i))
                    fitlog.add_best_metric({'test_acc_calib_mean': {'top_{}'.format(j + 1): 100*sum(tmp_acc_list)/len(tmp_acc_list)}}, name='iter_{}'.format(i))

        # 这里开始验证重要性和多样性的分数

        new_iter_example_idxs_heuristic_score = []
        now_iter_example_idxs_heuristic_score_importance = []
        now_iter_example_idxs_heuristic_score_diversity = []

        for example_idxs in new_iter_example_idxs_list:
            tmp_score_dict = calculate_example_idxs_score(feature_mean, feature_sim, example_idxs, args)

            new_iter_example_idxs_heuristic_score.append(tmp_score_dict['total'])
            now_iter_example_idxs_heuristic_score_importance.append(tmp_score_dict['importance'])
            now_iter_example_idxs_heuristic_score_diversity.append(tmp_score_dict['diversity'])

        example_idxs_topk_heuristic_score = torch.tensor(new_iter_example_idxs_heuristic_score)[
            example_idxs_topk_indices].tolist()
        example_idxs_topk_heuristic_score_importance = torch.tensor(now_iter_example_idxs_heuristic_score_importance)[
            example_idxs_topk_indices].tolist()
        example_idxs_topk_heuristic_score_diversity = torch.tensor(now_iter_example_idxs_heuristic_score_diversity)[
            example_idxs_topk_indices].tolist()

        logger.info('iteration_{} example_idxs_topk_heuristic_scores:{}'.format(i, example_idxs_topk_heuristic_score))
        logger.info('iteration_{} example_idxs_topk_heuristic_scores_importance:{}'.format(i,
                                                                                           example_idxs_topk_heuristic_score_importance))
        logger.info('iteration_{} example_idxs_topk_heuristic_scores_diversity:{}'.format(i,
                                                                                          example_idxs_topk_heuristic_score_diversity))

        logger.info('as compared, show three random example idxs heuristic score')

        for j in range(3):

            if args.label_balance:
                n_classes = N_LABELS_DICT[args.task]
                tmp_example_idxs = []
                for k in range(n_classes):
                    tmp_example_idxs.extend(
                        random.sample(label_to_candidate_idx[str(k)], args.candidate_example_num_every_label)
                    )
                example_idxs = tmp_example_idxs
            else:

                example_idxs = random.sample(list(range(candidate_num)), args.candidate_example_num_total)

            logger.info('random_example_{}'.format(j))
            logger.info(calculate_example_idxs_score(feature_mean, feature_sim, example_idxs, args))

        print('\n', '*' * 40, '\n')

    logger.info('total_calculate_count:{}'.format(total_calculate_count))

    logger.info('final_example_idxs_list:{}'.format(now_example_idxs_list))

    if args.num_iteration == 0:
        logger.info('start random sampling data to test acc')

        for i in range(10):
            random_example_idxs_list = []
            for j in range(args.beam_size * args.beam_size):
                if args.label_balance:
                    n_classes = N_LABELS_DICT[args.task]
                    tmp_example_idxs = []
                    for i in range(n_classes):
                        tmp_example_idxs.extend(
                            random.sample(label_to_candidate_idx[str(i)], args.candidate_example_num_every_label)
                        )
                    example_idxs = tmp_example_idxs
                else:

                    example_idxs = random.sample(list(range(candidate_num)), args.candidate_example_num_total)

                random_example_idxs_list.append(example_idxs)

            example_idxs_list_acc = verify_example_idxs_list_on_indication_data(candidate_data,
                                                                                random_example_idxs_list,
                                                                                indication_examples_from_train,
                                                                                args,
                                                                                tokenizer,
                                                                                model
                                                                                )['acc']
            if i == 0:
                logger.info('zero_shot_acc:{}'.format(example_idxs_list_acc[-1]))

            example_idxs_list_acc = example_idxs_list_acc[:-1]
            logger.info('sorted random examples acc at random trial {} : {}'.format(i, sorted(example_idxs_list_acc,
                                                                                              reverse=True)))
    fitlog.finish()

    # max_random_score = - 500
    # for i in range(20000):
    #     example_idxs = random.sample(list(range(candidate_num)), k=args.candidate_example_num_total)
    #
    #     tmp_score = example_idxs_score(feature_mean, feature_sim, example_idxs, args)['total']
    #     max_random_score = max(max_random_score, tmp_score)
    # logger.info('max_random_score:{}'.format(max_random_score))
    #
    # logger.info('feature_similarity_score[:3,:3]:{}'.format(feature_sim[:3, :3]))
    # logger.info('feature_mean[:3]:{}'.format(feature_mean[:3]))

    # parser.add_argument('--initial_examples_fp',required=True)

    123
