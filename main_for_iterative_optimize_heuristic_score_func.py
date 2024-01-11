import argparse
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

N_LABELS_DICT = {"SST-2": 2, "sst-5": 5, "mr": 2, "cr": 2, "mpqa": 2,
                 "subj": 2, "trec": 6, "CoLA": 2,
                 "amazon": 5, "yelp_full": 5, "yelp_binary": 2,
                 "agnews": 4, "copa": 2, "boolq": 2,
                 "RTE": 2, "cb": 3,
                 "yahoo": 10, "dbpedia": 14, 'snli': 3, 'mnli': 3, 'qnli': 2, 'wnli': 2, 'amazon_b':2}

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


def example_idxs_score(whole_feature_mean, whole_feature_sim, example_idxs, args):
    # logger.info('in example_idxs_score, example_idxs:{}'.format(example_idxs))
    tmp_feature_mean = whole_feature_mean[example_idxs]
    tmp_feature_sim = whole_feature_sim[example_idxs][:, example_idxs]

    tmp_feature_mean_score = torch.mean(tmp_feature_mean)
    tmp_diversity_score = - torch.mean(tmp_feature_sim)
    tmp_feature_score = tmp_feature_mean_score + tmp_diversity_score * args.diversity_score_scale

    return {'total': tmp_feature_score, 'importance': tmp_feature_mean_score, 'diversity': tmp_diversity_score}


def iterative_optimize_heuristic_score_function(args, feature_mean, feature_sim, initial_size_scale=1,print_iteration_info=False,label_to_candidate_idx=None,candidate_y=None):
    history_example_idxs_and_score_dict = {}
    initial_example_idxs_list = []
    candidate_num = feature_mean.size(0)
    for i in range(args.beam_size*initial_size_scale):
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

    total_calculate_count = 0
    now_example_idxs_list = initial_example_idxs_list

    total_calculate_count = 0

    for i in range(args.num_iteration):
        new_iter_example_idxs_list = []
        for j, example_idxs in enumerate(now_example_idxs_list):
            example_idxs_set = set(example_idxs)
            for k in range(args.beam_size):
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

                result_example_idxs = copy.deepcopy(example_idxs)
                result_example_idxs[tmp_replace_pos] = choosed_example_idx

                new_iter_example_idxs_list.append(result_example_idxs)

        new_iter_example_idxs_list.extend(now_example_idxs_list)
        new_iter_example_idxs_list = list(map(lambda x: tuple(sorted(x)), new_iter_example_idxs_list))
        new_iter_example_idxs_list = list(set(new_iter_example_idxs_list))
        new_iter_example_idxs_list = list(map(lambda x: list(x), new_iter_example_idxs_list))
        new_iter_example_idxs_list_with_seqidx = list(enumerate(new_iter_example_idxs_list))

        total_calculate_count += len(new_iter_example_idxs_list_with_seqidx)

        new_iter_example_idxs_seqscore = []
        now_iter_example_idx_seqscore_importance = []
        now_iter_example_idx_seqscore_diversity = []
        for j, example_idxs in new_iter_example_idxs_list_with_seqidx:
            tmp_score_dict = example_idxs_score(feature_mean, feature_sim, example_idxs, args)

            history_example_idxs_and_score_dict[tuple(example_idxs)] = tmp_score_dict['total']

            new_iter_example_idxs_seqscore.append(tmp_score_dict['total'])
            now_iter_example_idx_seqscore_importance.append(tmp_score_dict['importance'])
            now_iter_example_idx_seqscore_diversity.append(tmp_score_dict['diversity'])

        new_iter_example_idxs_seqscore = torch.tensor(new_iter_example_idxs_seqscore)
        example_idxs_topk = torch.topk(new_iter_example_idxs_seqscore, k=args.beam_size)

        example_idxs_topk_indices = example_idxs_topk.indices.tolist()
        example_idxs_topk_scores = example_idxs_topk.values.tolist()

        example_idxs_topk_importance = torch.tensor(now_iter_example_idx_seqscore_importance) \
            [example_idxs_topk_indices].tolist()
        example_idxs_topk_diversity = torch.tensor(now_iter_example_idx_seqscore_diversity) \
            [example_idxs_topk_indices].tolist()
        if print_iteration_info:
            if i == 0:
                logger.info('iteration_{} example_idxs_topk_scores:{}'.format(i, example_idxs_topk_scores[:args.beam_size]))
                logger.info('iteration_{} example_idxs_topk_importance:{}'.format(i, example_idxs_topk_importance[:args.beam_size]))
                logger.info('iteration_{} example_idxs_topk_diversity:{}'.format(i, example_idxs_topk_diversity[:args.beam_size]))
            else:
                logger.info('iteration_{} example_idxs_topk_scores:{}'.format(i, example_idxs_topk_scores))
                logger.info('iteration_{} example_idxs_topk_importance:{}'.format(i, example_idxs_topk_importance))
                logger.info('iteration_{} example_idxs_topk_diversity:{}'.format(i, example_idxs_topk_diversity))
            print('\n', '*' * 40, '\n')

        now_example_idxs_list = []

        for indice in example_idxs_topk_indices:
            now_example_idxs_list.append(new_iter_example_idxs_list[indice])

    new_iter_example_idxs_seqscore = new_iter_example_idxs_seqscore.tolist()

    tmp = list(zip(new_iter_example_idxs_seqscore, new_iter_example_idxs_list))
    tmp.sort(key=lambda x:x[0],reverse=True)

    new_iter_example_idxs_seqscore, new_iter_example_idxs_list = zip(*tmp)


    return new_iter_example_idxs_seqscore, new_iter_example_idxs_list, history_example_idxs_and_score_dict


if __name__ == '__main__':
    pass
