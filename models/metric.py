
import torch

from typing import Optional, Callable

from fastNLP import Metric

from .utils import find_backbone_seq_in_min_edit_distance


class IntegrateAccuracy(Metric):
    """
    直接计算句子级别、字、以及各种错误种类级别的准确率；

    """
    def __init__(self, predict_process_fn: Optional[Callable] = None, target_process_fn: Optional[Callable] = None,
                 backend="torch", aggregate_when_get_metric=None):
        super(IntegrateAccuracy, self).__init__(backend=backend, aggregate_when_get_metric=aggregate_when_get_metric)
        self.register_element(name='sent_correct', value=0, aggregate_method='sum', backend=backend)
        self.register_element(name='sent_total', value=0, aggregate_method="sum", backend=backend)

        self.register_element(name='tp', value=0, aggregate_method='sum', backend=backend)
        self.register_element(name='fp', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='fn', value=0, aggregate_method="sum", backend=backend)

        # 区分汉字拼音 right 和 wrong 的分别的准确率（注意这里实际上是 recall）；
        self.register_element(name='right_correct', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='right_total', value=0, aggregate_method="sum", backend=backend)

        self.register_element(name='wrong_correct', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='wrong_total', value=0, aggregate_method="sum", backend=backend)

        self.predict_process_fn = predict_process_fn if predict_process_fn is not None else lambda x: x
        self.target_process_fn = target_process_fn if target_process_fn is not None else lambda x: x

        for i in range(1, 7):
            self.register_element(name=f'{i}_correct', value=0, aggregate_method="sum", backend=backend)
            self.register_element(name=f'{i}_total', value=0, aggregate_method="sum", backend=backend)

    def update(self, preds, targets, mask, seq_len):
        """
        pinyin: 正确的拼音；
        seq_len: 指 target 的长度；

        """
        preds = preds.cpu()
        targets = targets.cpu()
        seq_len = seq_len.cpu() if isinstance(seq_len, torch.Tensor) else seq_len

        bsz = preds.shape[0]
        for i in range(bsz):
            processed_pred = self.predict_process_fn(preds[i], seq_len[i])
            processed_target = self.target_process_fn(targets[i], seq_len[i])
            right_charac = set(find_backbone_seq_in_min_edit_distance(processed_target, processed_pred)[0])

            if mask is not None:
                cur_mark = []
                for w in mask[i]:
                    if w != -1:
                        cur_mark.append(w)
                    else:
                        break
                assert len(processed_target) == len(cur_mark)
                for j in range(len(cur_mark)):
                    if cur_mark[j] == 0:
                        if j in right_charac:
                            self.right_correct += 1
                        self.right_total += 1
                    else:
                        if j in right_charac:
                            self.wrong_correct += 1
                            attr = getattr(self, f"{cur_mark[j]}_correct")
                            setattr(self, f"{cur_mark[j]}_correct", attr + 1)

                        self.wrong_total += 1
                        attr = getattr(self, f"{cur_mark[j]}_total")
                        setattr(self, f"{cur_mark[j]}_total", attr + 1)

            c = len(right_charac)
            self.tp += c
            self.fp += len(processed_pred) - c
            self.fn += len(processed_target) - c

            if isinstance(processed_pred, torch.Tensor):
                if processed_pred.tolist() == processed_target.tolist():
                    self.sent_correct += 1
            else:
                if processed_pred == processed_target:
                    self.sent_correct += 1
            self.sent_total += 1

    def get_metric(self) -> dict:
        tp = self.tp.get_scalar()
        fp = self.fp.get_scalar()
        fn = self.fn.get_scalar()
        right_correct = self.right_correct.get_scalar()
        right_total = self.right_total.get_scalar()
        wrong_correct = self.wrong_correct.get_scalar()
        wrong_total = self.wrong_total.get_scalar()
        assert tp + fn == right_total + wrong_total
        pre = tp / (tp + fp) if tp + fp != 0 else 0
        rec = tp / (tp + fn)
        res = {'f1': 2 * pre * rec / (pre + rec) if pre+rec != 0 else 0, 'pre': pre, 'rec': rec, 'tp': tp, 'fp': fp, 'fn': fn,
                'right_rec': right_correct / right_total if right_total != 0 else 0, 'right_correct': right_correct, 'right_total': right_total,
                'wrong_rec': wrong_correct / wrong_total if wrong_total != 0 else 0, 'wrong_correct': wrong_correct, 'wrong_total': wrong_total,
                'sent_acc': self.sent_correct.get_scalar() / self.sent_total.get_scalar(), 'sent_correct': self.sent_correct.get_scalar(),
                'sent_total': self.sent_total.get_scalar()}

        for i in range(1, 7):
            res[f"{i}_correct"] = getattr(self, f"{i}_correct").get_scalar()
            res[f"{i}_total"] = getattr(self, f"{i}_total").get_scalar()
            res[f'{i}_type_acc'] = res[f"{i}_correct"] / res[f"{i}_total"] if res[f"{i}_total"] != 0 else 0

        return res

#
# class BoundaryAccuracyPinyin(Metric):
#     """
#     计算 boundary predict 的准确率；
#     每一个输入都对应一个输出，如果一个字对应的全部字母的输出都正确才算这个字的 boundary predict 正确；
#     """
#     def __init__(self, backend="torch", aggregate_when_get_metric=None):
#         super(BoundaryAccuracyPinyin, self).__init__(backend=backend, aggregate_when_get_metric=aggregate_when_get_metric)
#
#         self.register_element(name='correct', value=0, aggregate_method="sum", backend=backend)
#         self.register_element(name='total', value=0, aggregate_method="sum", backend=backend)
#
#         self.register_element(name='sent_correct', value=0, aggregate_method="sum", backend=backend)
#         self.register_element(name='sent_total', value=0, aggregate_method="sum", backend=backend)
#
#     def update(self, preds, targets, boundary_sub_pinyin_len, seq_len):
#         for i in range(len(preds)):
#             _pred = preds[i]  # 这里就是原本的预测；
#             target = [w for w in targets[i] if w != -1]
#             pred = []
#             start_index = 1  # 跳过 cls；
#             for syllable_len in boundary_sub_pinyin_len[i]:
#                 if syllable_len != -1:
#                     pred.append(start_index)
#                     start_index += syllable_len
#                 else:
#                     break
#
#             all_true = True
#             start_index = 0
#             for word_len in seq_len[i]:
#                 if word_len != -1:
#                     tmp_pred = pred[start_index: start_index + word_len]
#                     tmp_target = target[start_index: start_index + word_len]
#                     if tmp_pred == tmp_target:
#                         self.correct += 1
#                     else:
#                         all_true = False
#                     self.total += 1
#                 else:
#                     break
#
#             if all_true:
#                 self.sent_correct += 1
#             self.sent_total += 1
#
#     def get_metric(self) -> dict:
#         correct = self.correct.get_scalar()
#         total = self.total.get_scalar()
#         sent_correct = self.sent_correct.get_scalar()
#         sent_total = self.sent_total.get_scalar()
#         return {
#             'acc': correct / total if total != 0 else 0, 'correct': correct, 'total': total,
#             'sent_acc': sent_correct / sent_total if sent_total != 0 else 0, 'sent_correct': sent_correct, 'sent_total': sent_total,
#         }


class CharacterBoundaryAccuracyPinyin(Metric):
    """
    计算 boundary predict 的准确率；
    每一个输入都对应一个输出，如果一个字对应的全部字母的输出都正确才算这个字的 boundary predict 正确；
    """
    def __init__(self, backend="torch", aggregate_when_get_metric=None):
        super(CharacterBoundaryAccuracyPinyin, self).__init__(backend=backend, aggregate_when_get_metric=aggregate_when_get_metric)

        self.register_element(name='correct', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='total', value=0, aggregate_method="sum", backend=backend)

        self.register_element(name='sent_correct', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='sent_total', value=0, aggregate_method="sum", backend=backend)

    def update(self, preds, targets, seq_len, boundary_sub_pinyin_len):

        for i in range(len(preds)):
            pred = preds[i][:seq_len[i]][1:-1]  # 去除 cls、sep 和 pad；
            target = targets[i][:seq_len[i]][1:-1]

            start_index = 0
            for syllable_len in boundary_sub_pinyin_len[i]:
                if syllable_len != -1:
                    tmp_syllable_pred = pred[start_index: start_index+syllable_len]
                    tmp_syllable_target = target[start_index: start_index+syllable_len]
                    if all(tmp_syllable_pred == tmp_syllable_target):
                        self.correct += 1
                    self.total += 1

                    start_index += syllable_len
                else:
                    break
            if all(pred==target):
                self.sent_correct += 1
            self.sent_total += 1

    def get_metric(self) -> dict:
        correct = self.correct.get_scalar()
        total = self.total.get_scalar()
        sent_correct = self.sent_correct.get_scalar()
        sent_total = self.sent_total.get_scalar()
        return {
            'acc': correct / total if total != 0 else 0, 'correct': correct, 'total': total,
            'sent_acc': sent_correct / sent_total if sent_total != 0 else 0, 'sent_correct': sent_correct, 'sent_total': sent_total,
        }









