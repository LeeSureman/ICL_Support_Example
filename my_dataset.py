from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from data import prepare_data
import logging
import torch

logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    # 这个dataset的candidate_demonstrations是元素为candidate example的list，它的大小的(num_candidate (+1) * num_indication)，candidate多的+1是zero shot的伪candidate
    # 上面是v4里的描述，这里v5又新加了一个特点，candidate_demonstrations也可以是candidate examples的list，也就是 以 “以candidate examples为元素的list” 为元素的list
    def __init__(self, args, candidate_demonstrations, indication_demonstrations, forward_method, tokenizer,
                 max_length_per_example, n_classes, template,
                 add_zero_shot_pseudo_candidate, candidate_demonstrations_element_check_tag, max_length=-1):
        self.candidate_demonstrations = candidate_demonstrations
        self.indication_demonstrations = indication_demonstrations

        self.candidate_num = len(self.candidate_demonstrations)
        self.indication_num = len(self.indication_demonstrations)

        self.forward_method = forward_method
        self.tokenizer = tokenizer

        self.n_classes = n_classes
        self.template = template
        self.args = args
        self.add_zero_shot_pseudo_candidate = add_zero_shot_pseudo_candidate
        self.candidate_demonstrations_element_check_tag = candidate_demonstrations_element_check_tag

        self.max_length_per_example = max_length_per_example

        if self.candidate_demonstrations_element_check_tag == 'list':
            assert max_length > max_length_per_example * 2
            self.mex_length = max_length
        else:
            self.max_length = max_length_per_example * 2

        if len(self.candidate_demonstrations) > 0:
            if self.candidate_demonstrations_element_check_tag == 'list':
                logger.info(
                    'self.candidate_demonstrations[0][0]: {}'.format(self.candidate_demonstrations[0][0]))

                assert type(self.candidate_demonstrations[0]) == type([]), type(self.candidate_demonstrations[0])
                assert type(self.candidate_demonstrations[0][0]) == type([]), type(self.candidate_demonstrations[0][0])
                assert type(self.candidate_demonstrations[0][0][0]) == type('123'), type(
                    self.candidate_demonstrations[0][0][0])

            elif self.candidate_demonstrations_element_check_tag == 'example':
                logger.info(
                    'self.candidate_demonstrations[0]: {}'.format(self.candidate_demonstrations[0]))

                assert type(self.candidate_demonstrations[0]) == type([]), type(self.candidate_demonstrations[0])
                assert type(self.candidate_demonstrations[0][0]) == type('123'), type(
                    self.candidate_demonstrations[0][0])

            else:
                raise NotImplementedError

        # assert self.forward_method in ['channel']

        logger.info('mydataset max_length_per_example:{}'.format(max_length_per_example))
        logger.info('mydataset max_length:{}'.format(max_length))

        pass

    def get_candidate_indication_hash(self):
        candidate_tuple = tuple(map(lambda x: tuple(x), self.candidate_demonstrations))
        indication_tuple = tuple(map(lambda x: tuple(x), self.indication_demonstrations))

        self_hash_result = hash(('candidate',) + candidate_tuple + ('indication',) + indication_tuple)
        return self_hash_result

    def __getitem__(self, item):
        # [candidate_num, indication_num, class_index]
        class_index = item % self.n_classes

        candidate_item = item // (len(self.indication_demonstrations) * self.n_classes)
        indication_item = (item // (self.n_classes)) % len(self.indication_demonstrations)

        # candidate = self.candidate_demonstrations[candidate_item]
        indication = self.indication_demonstrations[indication_item]

        if candidate_item < self.candidate_num:
            candidate = self.candidate_demonstrations[candidate_item]
            if self.candidate_demonstrations_element_check_tag == 'example':
                candidate = [candidate]
            result_n_classes = prepare_data(
                self.tokenizer, candidate, [indication],
                self.max_length_per_example * 2, self.max_length_per_example,
                self.n_classes,
                self.template,
                self.forward_method, use_demonstrations=True,warning_truncated=False)
        else:
            result_n_classes = prepare_data(
                self.tokenizer, [], [indication],
                self.max_length_per_example * 2, self.max_length_per_example,
                self.n_classes,
                self.template,
                self.forward_method, use_demonstrations=False,warning_truncated=False)

        # logger.info("result_n_classes:{}".format(result_n_classes))

        result = result_n_classes[class_index]
        result = {k: v[0] for k, v in result.items()}

        # logger.info(result.keys())
        # for k in result:
        #     logger.info('{}:{}'.format(k,result[k].size()))

        # result[]
        # result['token_type_ids'] = torch.tensor(result['token_type_ids'])
        # result['attention_mask'] = torch.tensor(result['attention_mask'])
        # result['input_ids'] = torch.tensor(result['input_ids'])

        result['candidate_index'] = torch.tensor(candidate_item, dtype=torch.long)
        result['indication_index'] = torch.tensor(indication_item, dtype=torch.long)
        result['class_index'] = torch.tensor(class_index, dtype=torch.long)

        return result

        pass

    def __len__(self):
        if self.add_zero_shot_pseudo_candidate:
            return (len(self.candidate_demonstrations) + 1) * len(self.indication_demonstrations) * self.n_classes
        else:
            return len(self.candidate_demonstrations) * len(self.indication_demonstrations) * self.n_classes
        # pass
