
import torch
from pathlib import Path
from dataclasses import is_dataclass
from typing import Dict, Union, List, Sequence
from copy import deepcopy

from fastNLP import Callback
import nvidia_smi


def extract_real_word_index(seq_len, sub_used_pinyin_len):
    """
    real_word_index 标记的是在原始数据中的 index；

    initials_index 和 other_index 标记的是 real_word_index 的 index；
    """

    real_word_index = []
    initials_index = []
    other_index = []

    sample_start_index = 0
    max_seq_len = max(seq_len)
    for i in range(len(seq_len)):
        begin_idx = i * max_seq_len
        cur_sample_index_list = [w + begin_idx for w in range(1, seq_len[i] - 1)]

        real_word_index.extend(cur_sample_index_list)

        start_index = sample_start_index
        for syllable_len in sub_used_pinyin_len[i]:
            syllable_len = int(syllable_len)
            if syllable_len != -1:
                initials_index.append(start_index)
                if syllable_len > 1:
                    other_index.extend(list(range(start_index+1, start_index+syllable_len)))

                start_index += syllable_len
            else:
                break

        sample_start_index += len(cur_sample_index_list)

    return real_word_index, initials_index, other_index


def predict_process_fn(pred, length):
    """
    去除 [cls], [sep], '-' 和 pad；'-': 6；
    pred: 一个 tensor: [1, real_length + pad_length]
    """

    return [w for w in pred[:length][1:-1] if w != 1]


def target_process_fn(target, length):
    return [w for w in target[:length][1:-1] if w != 1]


class EmpetyCudaCache(Callback):
    def on_train_epoch_end(self, trainer):
        torch.cuda.empty_cache()


class LRSchedCallback(Callback):
    def __init__(self, scheduler, start_epochs: int = 0, step_on: str = 'batch', name: str = "LRSchedCallback"):
        """
        根据 step_on 参数在合适的时机调用 scheduler 的 step 函数。

        :param scheduler: 实现了 step() 函数的对象
        :param step_on: 可选 ['batch'， 'epoch'] 表示在何时调用 scheduler 的 step 函数
        """
        assert hasattr(scheduler, 'step') and callable(scheduler.step), "The scheduler object should have a " \
                                                                        "step function."
        self.scheduler = scheduler
        self.start_epochs = start_epochs
        self.step_on = 0 if step_on == 'batch' else 1
        self.name = name

    def on_after_optimizers_step(self, trainer, optimizers):
        if self.step_on == 0:
            if self.start_epochs <= trainer.cur_epoch_idx:
                self.scheduler.step()

    def on_train_epoch_end(self, trainer):
        if self.step_on == 1:
            if self.start_epochs <= trainer.cur_epoch_idx:
                self.scheduler.step()

    def callback_name(self):
        return self.name


def make_cache_name(origin_name, *post_fix):
    origin_name = Path(origin_name)
    stem = origin_name.stem
    suffix = origin_name.suffix
    for other_fix in post_fix:
        other_fix = str(other_fix)
        if other_fix != '':
            stem = stem + "_" + other_fix
    origin_name = origin_name.parent.joinpath(stem + suffix)
    return origin_name


def dataclass_to_dict(data: "dataclasses.dataclass") -> Dict:
    """
    将传入的 ``dataclass`` 实例转换为字典。
    """
    if not is_dataclass(data):
        raise TypeError(f"Parameter `data` can only be `dataclass` type instead of {type(data)}.")
    _dict = dict()
    for _key in data.__dataclass_fields__:
        _dict[_key] = getattr(data, _key)
    return _dict


def find_backbone_seq_in_min_edit_distance(label: Union[str, List, torch.LongTensor],
                                      predict: Union[str, List, torch.LongTensor]):
    assert type(label) == type(predict)

    min_edit_path = {(0, 0): [("replace", None)]}

    min_distance_matrix = [[0] * (len(predict) + 1) for _ in range(len(label) + 1)]
    for i in range(1, len(predict)+1):
        min_distance_matrix[0][i] = i
        min_edit_path[(0, i)] = [("delete", predict[i-1])]

    for i in range(1, len(label)+1):
        min_distance_matrix[i][0] = i
        min_edit_path[(i, 0)] = [("add", label[i-1])]

        for j in range(1, len(predict)+1):
            one_or_zero = 0 if label[i-1] == predict[j-1] else 1
            distance = {
                'delete': min_distance_matrix[i][j-1] + 1,
                'add': min_distance_matrix[i-1][j] + 1,
                'replace': min_distance_matrix[i-1][j-1] + one_or_zero
            }
            operation = min(distance, key=distance.get)
            min_distance_matrix[i][j] = distance[operation]
            from_who = {
                'delete': (i, j-1),
                'add': (i-1, j),
                'replace': (i-1, j-1)
            }
            real_operation = "zero_replace" if operation == "replace" and one_or_zero == 0 else operation
            # print((i, j), operation)
            # print(min_edit_path)
            # print(min_edit_path[from_who[operation]])
            min_edit_path[(i, j)] = deepcopy(min_edit_path[from_who[operation]])
            min_edit_path[(i, j)].append((real_operation, i-1))

    backbone_seq = []
    for operation in min_edit_path[(len(label), len(predict))]:
        if operation[0] == "zero_replace":
            backbone_seq.append(operation[1])

    return backbone_seq, min_distance_matrix[-1][-1]


def check_gpu_available(device: Union[int, List[int]], limit=18 * 1024**3):
    nvidia_smi.nvmlInit()

    if not isinstance(device, Sequence):
        device = [device]

    could_use = True
    for each_device in device:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(each_device)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        if info.free < limit:
            could_use = False
            break

    return could_use