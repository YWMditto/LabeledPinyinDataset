import json
import random
from copy import deepcopy
from functools import reduce
import jieba
jieba.initialize()
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import torch

from fastNLP import logger, DataSet


def load_simulated_data(filepath, wrong_sent_ratio, overall_typo_ratio, typo_type_dist, length=-1):
    """
    加载 train.txt 或者 dev.txt，这些数据是通过 pypinyin 批量生成，而不是人工标注的数据；

    对于训练数据，我们需要模拟真实标注下的错误情况，来对原本正确的训练数据添加 '错误拼音' 的噪声；
    该函数会首先确定总共有多少句子（输入的拼音）会添加噪声（wrong_sent_ratio），然后参数 overall_type_ratio 会设定一个句子中出现1、2、3个错误的
     句子数量的比例，例如 (0.72, 0.2, 0.08) 表示在含有错误的句子中，72% 的句子只含有 1 个错误，20% 的句子含有 2 个，8% 的句子含有 3 个。
    然后参数 typo_type_dist 会控制总共出现的错误种类的数量的比例，总共的错误类别分别如下所示：

    该函数返回一个 list，其中每个对象为：
        1. 如果 wrong_sent_ratio != 0:
            (domain, cn, pinyin, used_pinyin, [(index, wrong_type), ...])
            其中 used_pinyin 表示实际训练中会使用的拼音序列，其可能是正确的，也可能是添加过噪声的，index 表示错误的拼音所表示的字在整句话中
             所在的位置， wrong_type 则对应着该拼音错误的种类；
        2. 否则:
            (domain, cn, pinyin, used_pinyin)

    a. 字母替换: hao -> hap
    b. 缺少字母: hao -> ho
    c. 额外字母: hao -> haop
    d. 字母顺序错误: hao -> hoa
    e. 只包含首字母: hao -> h
    f. 其余错误（混合）: shang -> shanh

    :param: filepath: 数据所在的文件位置；
    :param: wrong_sent_ratio: e.g. 0.35
    :param: overall_type_ratio: e.g. (0.72, 0.2, 0.08)；
    :param: typo_type_dist: e.g. (0.24, 0.16, 0.31, 0.1, 0.15, 0.04)

    :return: (domain, cn, pinyin, used_pinyin, [(index, wrong_type), ...]) or (domain, cn, pinyin, used_pinyin)
    """

    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    if length != -1:
        data = data[:length]

    using_data = []
    # 先确定哪些数据是错误数据；
    if wrong_sent_ratio != 0:
        logger.info("融合错误数据！")
        assert abs(sum(overall_typo_ratio) - 1) < 0.000001 and abs(sum(typo_type_dist) - 1) < 0.000001

        random.shuffle(data)
        overall_typo_num = int(len(data) * wrong_sent_ratio)

        # 先确定一个句子中错误数量超过 1 的句子，错误数据依次为错误数量为 3、2、1 的句子；
        each_len_num = [int(overall_typo_num * overall_typo_ratio[2]), int(overall_typo_num * overall_typo_ratio[1])]
        each_len_num.append(overall_typo_num - sum(each_len_num))

        # 需要维持一个至今已经构造多少错误的字典；
        already_made_typo = dict((w, 0) for w in range(1, 7))
        all_word_typo_num = 3 * each_len_num[0] + 2 * each_len_num[1] + each_len_num[2]
        each_typo_type_num = [int(all_word_typo_num * w) for w in typo_type_dist[:-1]]
        each_typo_type_num.append(all_word_typo_num - sum(each_typo_type_num))
        each_typo_type_num = dict((w+1, each_typo_type_num[w]) for w in range(6))

        logger.info(f"您选择了这样的错误分布：{each_typo_type_num}.")

        def merge_typo_into_one_sent(sample, typo_num: int):
            domain = sample[0]
            pinyin = sample[2]
            wrong_pinyin_list = sample[3:]
            wrong_pinyin_indexes = list(range(1, 7))
            random.shuffle(wrong_pinyin_indexes)

            merged_wrong_pinyin = deepcopy(pinyin)
            merged_typo_index = set()
            added_typo = []
            for j in wrong_pinyin_indexes:
                # 此时 j 就是真实的 type 的类别；
                wrong_pinyin = wrong_pinyin_list[j-1]
                for i in range(len(wrong_pinyin)):
                    if wrong_pinyin[i] != pinyin[i]:
                        merged_typo_index.add(i)
                        merged_wrong_pinyin[i] = wrong_pinyin[i]
                        # 更新已经添加的错误数据的错误分布；
                        already_made_typo[j] += 1
                        should_add = True

                        for index in range(len(added_typo)):
                            if added_typo[index][0] == i:
                                added_typo[index] = (i, j)
                                should_add = False
                                break
                        if should_add:
                            added_typo.append((i, j))

                        break
                if len(merged_typo_index) == typo_num:
                    break
            return (domain, sample[1], sample[2], merged_wrong_pinyin, added_typo)

        # 先处理一个句子中错误数量为 3、2 的数据；
        start_index = 0
        for i in range(2):
            # 把前 overall_typo_num 的数据使用错误数据；
            for sample in data[start_index:start_index+each_len_num[i]]:
                using_data.append(merge_typo_into_one_sent(sample, (3, 2)[i]))
            start_index = start_index + each_len_num[i]

        # 再根据剩余的每一个 typo 种类可添加的数量进行添加；
        start_index = sum(each_len_num[:2])
        for typo_type in already_made_typo:
            rest_num = each_typo_type_num[typo_type] - already_made_typo[typo_type]
            for sample in data[start_index:start_index+rest_num]:
                # 找到错误字的index；
                added_typo = None
                for i in range(len(sample[2])):
                    if sample[2][i] != sample[2+typo_type][i]:
                        added_typo = [(i, typo_type)]
                        break
                # 可能有些错误加到原拼音上是没有效果的，例如单个字母的字的首字母；
                if added_typo is not None:
                    using_data.append((sample[0], sample[1], sample[2], sample[2+typo_type], added_typo))
            start_index += rest_num
        logger.info(f"错误的句子的数量一共为：{len(using_data)}.")

        for sample in data[len(using_data):]:
            using_data.append((sample[0], sample[1], sample[2], sample[2]))
    else:
        logger.info("全部使用正确数据！")
        for sample in data:
            using_data.append((sample[0], sample[1], sample[2], sample[2]))

    logger.info(f"一共加载 {len(using_data)} 条数据.")
    return using_data


def load_labeled_data(filepath):
    """
    加载 test.txt，人工真实标注的数据；
    """
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)

    using_data = []
    for sample in data:
        using_data.append((sample["domain"], sample['cn'], sample['pinyin'], sample['labeled_pinyin'], sample['typo_type']))

    logger.info(f"标注数据一共 {len(using_data)} 条.")
    return using_data


def add_blank_tokens_in_cn(pinyin, cn, blank_token, add_start: bool = True):
    try:
        assert len(pinyin) == len(cn)
        res = []
        for i in range(len(pinyin)):
            sub_pinyin = pinyin[i]
            sub_res = [blank_token] * len(sub_pinyin)
            if add_start:
                sub_res[0] = cn[i]
            else:
                sub_res[-1] = cn[i]
            res.extend(sub_res)
        return res
    except Exception as e:
        print(pinyin)
        print(cn)
        raise e


def genrerate_split_boundary(pinyin, split_cn):
    """
    上海 北京
    shanghai beijing
    10000000 1000000
    """

    res = [0] * sum(len(w) for w in pinyin)

    split_cn_pinyin_len = []
    start_index = 0
    charac_idnex = 0
    for i in range(len(split_cn)):
        res[start_index] = 1
        cur_word = split_cn[i]
        for j in range(len(cur_word)):
            start_index += len(pinyin[charac_idnex+j])
        charac_idnex += len(cur_word)
        split_cn_pinyin_len.append(start_index - sum(split_cn_pinyin_len))
    return res, split_cn_pinyin_len


def generate_character_boundary(pinyin):
    res = [0] * sum(len(w) for w in pinyin)

    start_index = 0
    for i in range(len(pinyin)):
        res[start_index] = 1
        start_index += len(pinyin[i])

    return res



def construct_word_level_attention_mask(pinyin, mode=0):
    """
    构造用于训练 teacher model 的 attention mask；
    例如对于[cls] fu dan [sep]：
           cls    f   u   d   a   n   sep

    cls     1     1   1   1   1   1    1

    f       1     1   1   1   0   0    1

    u       1     1   1   0   0   0    1

    d       1     1   0   1   1   1    1

    a       1     0   0   1   1   1    1

    n       1     0   0   1   1   1    1

    sep     1     1   1   1   1   1    1

    添加时两个步骤，首先根据每一个sample生成自己的mask；
    然后在一个 batch 中再添加 pad 的mask；
    :param: pinyin: ['fu', 'dan']
    """

    pinyin_length = sum(len(w) for w in pinyin) + 2

    if mode == 0:
        # cls 和 sep 的 mask；
        mask = torch.zeros((pinyin_length, pinyin_length))
        mask[[0, -1], :] = 1
        mask[:, [0, -1]] = 1

        # 首字母的 mask；
        begin_index = 1
        initial_index = []
        for word in pinyin:
            initial_index.append(begin_index)
            # 每一个字自己的拼音字母的 mask；
            mask[begin_index: begin_index+len(word), begin_index: begin_index+len(word)] = 1
            begin_index += len(word)

        mask[[[w] for w in initial_index], [initial_index for _ in range(len(initial_index))]] = 1

        return mask
    else:
        return torch.ones((pinyin_length, pinyin_length))


def construct_multiple_attention_mask(split_cn, pinyin, mode=0):
    """
    根据 cn 和 pinyin 来构造 multiple attention mask，用于让模型将更多地注意力集中在当前的这个词身上，相当于我们已经分好了词然后再让
    模型去做 p2c；
    """
    pinyin_length = sum(len(w) for w in pinyin) + 2
    if mode == 0:
        attention_mask = torch.ones((pinyin_length, pinyin_length))
        start_index = 1
        charac_index = 0
        for i in range(len(split_cn)):
            cur_word = split_cn[i]
            indexes = []
            for j in range(len(cur_word)):
                indexes.append(start_index)
                start_index += len(pinyin[charac_index+j])
            charac_index += len(cur_word)

            row, col = list(zip(*list(product(indexes, indexes))))
            attention_mask[row, col] = 2
    elif mode == 1:
        # cls 和 sep 的 mask；
        attention_mask = torch.ones((pinyin_length, pinyin_length))
        start_index = 1
        end_index = 1
        charac_index = 0
        for i in range(len(split_cn)):
            cur_word = split_cn[i]
            for j in range(len(cur_word)):
                end_index += len(pinyin[charac_index + j])
            charac_index += len(cur_word)

            attention_mask[start_index: end_index, start_index: end_index] = 2
            start_index = end_index
    else:
        raise ValueError

    return attention_mask


@dataclass
class DataConfig:
    blank_token: str = "[unused1]"  # 因为我们是直接输入拼音序列，输出汉字，但是拼音序列通常长于汉字，因此我们只让拼音的首字母输出汉字，其余位置输出 blank_token；
    attention_mode: int = 0  # 用于训练理论上界模型时会被使用，生成包含分词信息的 attention mask；


def prepare_dataset(data, tokenizer, data_config: DataConfig, num_proc: int=8, use_labeled_pinyin: bool = True):
    """
    根据传入的数据生成用于训练的数据，主要过程涉及到 tokenize 的操作，例如将输入的 pinyin 序列转换成对应的 token id，将输出的汉字序列同样
     转换成 token id；
    """
    assert hasattr(data_config, "theoretical_training")

    tokenizer.add_tokens(data_config.blank_token, special_tokens=True)

    dataset = {
        'idx': [],
        'domain': [],
        'cn': [],
        'pinyin': [],
        'used_pinyin': [],
        'typo': []
    }

    for idx, sample in enumerate(data):
        if len(sample) == 4:
            domain, cn, pinyin, used_pinyin = sample
            added_typo = []
        else:
            domain, cn, pinyin, used_pinyin, added_typo = sample
        dataset['idx'].append(idx)
        dataset['domain'].append(domain)
        dataset['cn'].append(cn)
        dataset['pinyin'].append(pinyin)
        if use_labeled_pinyin:
            dataset['used_pinyin'].append(used_pinyin)
        else:
            dataset['used_pinyin'].append(pinyin)
        dataset['typo'].append(added_typo)
    dataset = DataSet(dataset)

    def _process_one_sample(batch):
        used_pinyin = batch['used_pinyin']
        cn = batch['cn']
        added_typo = batch['typo']

        split_cn = list(jieba.cut(cn))

        # used_pinyin
        used_pinyin_token = tokenizer(list("".join(used_pinyin)), add_special_tokens=False)
        used_pinyin_token["input_ids"] = [tokenizer.cls_token_id] + reduce(lambda x, y: x + y,
                                         used_pinyin_token["input_ids"]) + [tokenizer.sep_token_id]

        # cn used_pinyin
        cn_after_used_pinyin_bt = add_blank_tokens_in_cn(used_pinyin, cn, data_config.blank_token)
        cn_after_used_pinyin_bt_token = tokenizer(cn_after_used_pinyin_bt, add_special_tokens=False)
        cn_after_used_pinyin_bt_token["input_ids"] = [tokenizer.cls_token_id] + reduce(lambda x, y: x + y,
                                                    cn_after_used_pinyin_bt_token["input_ids"]) + [tokenizer.sep_token_id]
        used_pinyin_split_boundary, used_pinyin_split_cn_pinyin_len = genrerate_split_boundary(used_pinyin, split_cn)
        used_pinyin_split_boundary = [2] + used_pinyin_split_boundary + [2]
        used_pinyin_character_boundary = [2] + generate_character_boundary(used_pinyin) + [2]

        # mask
        # mask 用于 metric 评测时输出不同 typo type 的转换准确率；
        # mask 的长度和 cn 一致（不加 cls 和 sep），0 表示这个位置的字没有错误，1 ~ 6 表示 typo type；
        mask = [0] * len(cn)
        for typo in added_typo:
            mask[typo[0]] = typo[1]

        returned_dict = {
            "used_pinyin": used_pinyin_token["input_ids"],
            'used_pinyin_len': len(used_pinyin_token['input_ids']),
            "sub_used_pinyin_len": [len(w) for w in used_pinyin],
            "used_pinyin_attention_mask": torch.ones([len(used_pinyin_token["input_ids"]), len(used_pinyin_token["input_ids"])]),

            # 用于让 model 学习到分词；
            "cn_used_pinyin": cn_after_used_pinyin_bt_token["input_ids"],

            'used_pinyin_split_boundary': used_pinyin_split_boundary,
            'used_pinyin_split_cn_pinyin_len': used_pinyin_split_cn_pinyin_len,
            # 长度和 used_pinyin_split_boundary 一样，同样需要在学习的时候需要去除 cls 和 sep 的位置；
            'used_pinyin_character_boundary': used_pinyin_character_boundary,

            'split_cn_len': [len(w) for w in split_cn],

            "mask": mask
        }

        if data_config.theoretical_training:
            used_pinyin_attention_mask = construct_word_level_attention_mask(used_pinyin, mode=data_config.attention_mode)
            used_pinyin_multiple_attention_mask = construct_multiple_attention_mask(split_cn, used_pinyin, mode=data_config.attention_mode)
            returned_dict['used_pinyin_attention_mask'] = used_pinyin_attention_mask
            returned_dict['used_pinyin_multiple_attention_mask'] = used_pinyin_multiple_attention_mask

        return returned_dict

    dataset.apply_more(_process_one_sample, num_proc=num_proc)
    dataset.delete_field("pinyin")
    return dataset


def collate_fn(batch, pad_token_id: int = 0):
    used_pinyin_max_length = max([len(x["used_pinyin"]) for x in batch])
    sub_used_pinyin_len_max_length = max([len(x["sub_used_pinyin_len"]) for x in batch])
    used_pinyin_attention_mask_max_length = max([len(x["used_pinyin_attention_mask"]) for x in batch])
    cn_used_pinyin_max_length = max([len(x["cn_used_pinyin"]) for x in batch])
    mask_max_length = max([len(x["mask"]) for x in batch])
    used_pinyin_split_cn_pinyin_len_max_length = max([len(x["used_pinyin_split_cn_pinyin_len"]) for x in batch])
    split_cn_len_max_length = max([len(x["split_cn_len"]) for x in batch])

    all_lists = defaultdict(list)
    for sample in batch:
        for key, value in sample.items():
            if key == "used_pinyin":
                all_lists[key].append(value + [pad_token_id] * (used_pinyin_max_length - len(value)))
            elif key == "used_pinyin_attention_mask":
                _tmp_mask = torch.cat([value, torch.zeros([len(value), used_pinyin_attention_mask_max_length - len(value)])],
                                      dim=-1)
                _tmp_mask = torch.cat([_tmp_mask, torch.zeros(
                    [used_pinyin_attention_mask_max_length - len(value), used_pinyin_attention_mask_max_length])], dim=0)
                all_lists[key].append(_tmp_mask.unsqueeze(0))
            elif key == "used_pinyin_multiple_attention_mask":
                _tmp_mask = torch.cat([value, torch.zeros([len(value), used_pinyin_attention_mask_max_length - len(value)])],
                                      dim=-1)
                _tmp_mask = torch.cat([_tmp_mask, torch.zeros(
                    [used_pinyin_attention_mask_max_length - len(value), used_pinyin_attention_mask_max_length])], dim=0)
                all_lists[key].append(_tmp_mask.unsqueeze(0))
            elif key == "cn_used_pinyin":
                all_lists["cn_used_pinyin"].append(value + [pad_token_id] * (cn_used_pinyin_max_length - len(value)))
                all_lists["cn_used_pinyin_len"].append(len(value))
            elif key == "sub_used_pinyin_len":
                all_lists[key].append(value + [-1] * (sub_used_pinyin_len_max_length - len(value)))
            elif key == "mask":
                all_lists[key].append(value + [-1] * (mask_max_length - len(value)))
            elif key == "idx":
                all_lists[key].append(value)
            elif key == "used_pinyin_split_boundary":
                all_lists['used_pinyin_split_boundary'].append(value + [-1] * (used_pinyin_max_length - len(value)))
            elif key == "used_pinyin_split_cn_pinyin_len":
                all_lists["used_pinyin_split_cn_pinyin_len"].append(
                    value + [-1] * (used_pinyin_split_cn_pinyin_len_max_length - len(value)))
            elif key == "used_pinyin_character_boundary":
                all_lists["used_pinyin_character_boundary"].append(value + [-1] * (used_pinyin_max_length - len(value)))
            elif key == "split_cn_len":
                all_lists["split_cn_len"].append(value + [-1] * (split_cn_len_max_length - len(value)))
            else:
                all_lists[key].append(value)

    all_lists['used_pinyin_attention_mask'] = torch.cat(all_lists['used_pinyin_attention_mask'], dim=0).long()
    if "used_pinyin_multiple_attention_mask" in all_lists:
        all_lists['used_pinyin_multiple_attention_mask'] = torch.cat(all_lists['used_pinyin_multiple_attention_mask'], dim=0)
    for key, value in all_lists.items():
        if key not in {"pinyin_multiple_attention_mask", "used_pinyin_multiple_attention_mask", "domain", "cn", "typo"}:
            try:
                all_lists[key] = torch.LongTensor(value)
            except:
                a = 1

    return dict(all_lists)



if __name__ == "__main__":

    # data = load_simulated_data(
    #     filepath="/remote-home/xgyang/pinyin/pinyin_ime/LabeledPinyinDataset/data/dev.txt",
    #     wrong_sent_ratio=0.35,
    #     overall_typo_ratio=(0.72, 0.2, 0.08),
    #     typo_type_dist=(0.24, 0.16, 0.31, 0.1, 0.15, 0.04),
    #     length=10
    # )
    #
    # from fastNLP.core import prepare_torch_dataloader
    # from transformers import BertTokenizer
    #
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    #
    # data_config = DataConfig()
    # data_config.theoretical_training = False
    #
    # dataset = prepare_dataset(
    #     data,
    #     tokenizer,
    #     data_config,
    #     num_proc=64
    # )
    # dataloader = prepare_torch_dataloader(dataset, batch_size=4, collate_fn=collate_fn)
    #
    #
    # for idx, batch in enumerate(dataloader):
    #
    #     if idx > 2:
    #         break
    #
    #
    #
    # cn_list = [
    #     '你好黑龙江',
    #     '上海北京',
    #     '他是一个男人'
    # ]
    # pinyin_list = [
    #     ['ni', 'hao', 'hei', 'long', 'jaing'],
    #     ['shang', 'hai', 'bei', 'jing'],
    #     ["ta", "shi", "yi", "ge", "nan", "ren"]
    # ]
    #
    # for i in range(len(pinyin_list)):
    #     split_cn = list(jieba.cut(cn_list[i]))
    #     print(split_cn)
    #
    #     # attention_mask = construct_word_level_attention_mask(pinyin_list[i], mode=0)
    #     multi_attention_mask = construct_multiple_attention_mask(split_cn, pinyin_list[i], mode=0)
    #     print(multi_attention_mask)
    #
    #     # print(generate_split_boundary(pinyin_list[i], split_cn))
    #     # print(generate_character_boundary(pinyin_list[i]))
    #
    #     a = 1




    test_data = load_labeled_data("/remote-home/xgyang/pinyin/pinyin_ime/LabeledPinyinDataset/data/test.txt")

    a = 1






















