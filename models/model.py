
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any

from .modified_bert import BertModel
from .utils import extract_real_word_index

from fastNLP import logger


@dataclass
class ModelConfig:
    pretrained_model: str = 'bert-base-chinese'
    boundary_predict: bool = False
    character_boundary_predict: bool = False
    use_lstm: bool = False
    adjust_loss_weight: bool = False
    loss_reduction: Optional[Any] = None

    def __post_init__(self):
        if self.loss_reduction is not None and self.loss_reduction != "equal":
            self.loss_reduction = eval(self.loss_reduction)


@dataclass
class ModelOutput:
    logits: Optional[torch.FloatTensor] = None
    character_boundary_logits: Optional[torch.FloatTensor] = None
    boundary_logits: Optional[torch.FloatTensor] = None  # multitask learning


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(Model, self).__init__()
        assert hasattr(model_config, 'theoretical_training')

        self.model_config = model_config

        self.pretrained_model = BertModel.from_pretrained(model_config.pretrained_model)
        self.pretrained_config = self.pretrained_model.config
        self.decode_layer = nn.Linear(self.pretrained_config.hidden_size, self.pretrained_config.vocab_size)

        if model_config.theoretical_training:
            logger.info('训练理论上界模型；')
            model_config.boundary_predict = False
            model_config.character_boundary_predict = False
        else:
            if model_config.boundary_predict:
                logger.info("注意当前使用了 boundary_predict；")
                self.boundary_layer = nn.Linear(self.pretrained_config.hidden_size, 2)

            if model_config.character_boundary_predict:
                logger.info("注意当前使用了 character_boundary_predict；")
                self.character_boundary_layer = nn.Linear(self.pretrained_config.hidden_size, 2)

        if model_config.use_lstm:
            logger.info("注意当前使用了 lstm；")
            self.lstm = nn.LSTM(input_size=self.pretrained_config.hidden_size, hidden_size=self.pretrained_config.hidden_size // 2,
                                num_layers=1, batch_first=True, bidirectional=True)

        if model_config.adjust_loss_weight:
            logger.info("注意当前使用了 adjust_loss_weight；")
            logger.info(f"当前使用的 loss reduction 是 {model_config.loss_reduction}；")
        else:
            if model_config.loss_reduction is not None:
                logger.info("注意当前没有使用 adjust loss weight，但是 loss reduction 不是 None；")

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        self.boundary_loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        multiple_attention_mask = None if not self.model_config.theoretical_training else batch['used_pinyin_multiple_attention_mask']

        former_former_outputs = self.pretrained_model(
            input_ids=batch["used_pinyin"],
            attention_mask=batch["used_pinyin_attention_mask"],
            multiple_attention_mask=multiple_attention_mask,

            output_hidden_states=True,
            forward_layers=3,
        )

        character_boundary_logits = None
        if self.model_config.character_boundary_predict:
            character_boundary_hidden_states = former_former_outputs.hidden_states[-1]
            character_boundary_logits = self.character_boundary_layer(character_boundary_hidden_states)

        former_outputs = self.pretrained_model(
            inputs_embeds=former_former_outputs.hidden_states[-1],
            attention_mask=batch["used_pinyin_attention_mask"],
            multiple_attention_mask=multiple_attention_mask,

            output_hidden_states=True,
            forward_layers=(3, 6),  # 左开右闭；
        )

        boundary_logits = None
        if self.model_config.boundary_predict:
            boundary_logits = self.boundary_layer(former_outputs.hidden_states[-1])

        outputs = self.pretrained_model(
            inputs_embeds=former_outputs.hidden_states[-1],
            attention_mask=batch["used_pinyin_attention_mask"],
            multiple_attention_mask=multiple_attention_mask,

            forward_layers=-6,
        )

        last_hidden_state = outputs.last_hidden_state
        if self.model_config.use_lstm:
            sent_length = batch['used_pinyin_len']
            packed_embs = pack_padded_sequence(last_hidden_state, sent_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_outs, (hidden, _) = self.lstm(packed_embs)
            last_hidden_state, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        logits = self.decode_layer(last_hidden_state)

        return ModelOutput(
            logits=logits,
            character_boundary_logits=character_boundary_logits,
            boundary_logits=boundary_logits,
        )

    def train_step(self, batch):
        model_output = self(batch)
        real_word_index, initials_index, other_index = extract_real_word_index(batch['used_pinyin_len'],
                                                                               batch["sub_used_pinyin_len"])
        logits = model_output.logits
        targets = batch["cn_used_pinyin"]
        logits = logits.view(-1, self.pretrained_config.vocab_size)
        targets = targets.view(-1)
        loss = self.loss_fn(logits[real_word_index], targets[real_word_index])  # 需要我们自己手动进行mean，因为会进行权重的调整；

        if not self.model_config.adjust_loss_weight:
            loss = torch.mean(loss)
        else:
            N = len(real_word_index)  # 所有 pinyin 字母；
            M = len(initials_index)
            initial_loss = loss[initials_index]
            other_loss = loss[other_index]
            if self.model_config.loss_reduction is None:
                # loss reduction 为 None 时采取只增加首字母 loss 对应句子切分的比例；
                # 对每一个字的首字母增加权重；
                loss = (1 / N + 1 / M) * torch.sum(initial_loss) + 1 / N * torch.sum(other_loss)
            elif isinstance(self.model_config.loss_reduction, int):
                # 为 int 时表示固定比例；
                loss = self.loss_reduction / N * torch.sum(initial_loss) + 1 / N * torch.sum(other_loss)
            elif self.model_config.loss_reduction == "equal":
                # equal 表示预测字所贡献的损失函数和预测 '-' 的损失偶函数的 token 的个数的比例是相同的；
                loss = (N - M) / (N * M) * torch.sum(initial_loss) + 1 / N * torch.sum(other_loss)
            else:
                raise RuntimeError

        # 计算额外的 multitask loss；
        if self.model_config.boundary_predict:
            boundary_logits = model_output.boundary_logits
            boundary_logits = boundary_logits.view(-1, 2)
            used_pinyin_split_boundary_targets = batch['used_pinyin_split_boundary']
            used_pinyin_split_boundary_targets = used_pinyin_split_boundary_targets.view(-1)
            loss += self.boundary_loss_fn(boundary_logits[real_word_index],
                                                   used_pinyin_split_boundary_targets[real_word_index])

        if self.model_config.character_boundary_predict:
            character_boundary_logits = model_output.character_boundary_logits
            character_boundary_logits = character_boundary_logits.view(-1, 2)
            used_pinyin_character_boundary = batch['used_pinyin_character_boundary']
            used_pinyin_character_boundary = used_pinyin_character_boundary.view(-1)
            loss += self.boundary_loss_fn(character_boundary_logits[real_word_index],
                                          used_pinyin_character_boundary[real_word_index])

        return {"loss": loss}

    def evaluate_step(self, batch):
        model_output = self(batch)
        logits = model_output.logits
        logits = torch.max(logits, dim=-1)[1]
        return {"preds": logits, "targets": batch["cn_used_pinyin"], 'seq_len': batch['cn_used_pinyin_len']}

    def evaluate_boundary_step(self, batch):
        assert self.model_config.boundary_predict
        multiple_attention_mask = None if not self.model_config.theoretical_training else batch[
            'used_pinyin_multiple_attention_mask']

        former_outputs = self.pretrained_model(
            input_ids=batch["used_pinyin"],
            attention_mask=batch["used_pinyin_attention_mask"],
            multiple_attention_mask=multiple_attention_mask,

            output_hidden_states=True,
            forward_layers=6,
        )
        boundary_logits = self.boundary_layer(former_outputs.hidden_states[-1])

        boundary_logits = torch.max(boundary_logits, dim=-1)[1]
        return {"preds": boundary_logits, "targets": batch["used_pinyin_split_boundary"],
                'seq_len': batch['cn_used_pinyin_len'], "boundary_sub_pinyin_len": batch['used_pinyin_split_cn_pinyin_len']}

    def evaluate_character_boundary_step(self, batch):
        assert self.model_config.character_boundary_predict
        multiple_attention_mask = None if not self.model_config.theoretical_training else batch[
            'used_pinyin_multiple_attention_mask']

        former_former_outputs = self.pretrained_model(
            input_ids=batch["used_pinyin"],
            attention_mask=batch["used_pinyin_attention_mask"],
            multiple_attention_mask=multiple_attention_mask,

            output_hidden_states=True,
            forward_layers=3,
        )
        character_boundary_hidden_states = former_former_outputs.hidden_states[-1]
        character_boundary_logits = self.character_boundary_layer(character_boundary_hidden_states)
        character_boundary_logits = torch.max(character_boundary_logits, dim=-1)[1]
        return {"preds": character_boundary_logits, "targets": batch["used_pinyin_character_boundary"],
                'seq_len': batch['cn_used_pinyin_len'], "boundary_sub_pinyin_len": batch['sub_used_pinyin_len']}
