import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import wandb
from copy import deepcopy

from transformers import BertTokenizerFast, HfArgumentParser

from fastNLP import Trainer, Evaluator, logger, cache_results, prepare_torch_dataloader, Event
from fastNLP.core.callbacks import CheckpointCallback, MoreEvaluateCallback
from fastNLP.core.drivers import torch_seed_everything

from models.model import Model, ModelConfig
from models.metric import IntegrateAccuracy, CharacterBoundaryAccuracyPinyin
from models.utils import predict_process_fn, target_process_fn, EmpetyCudaCache, LRSchedCallback, make_cache_name, dataclass_to_dict
from models.data_preprocess import DataConfig, load_simulated_data, load_labeled_data, prepare_dataset, collate_fn


@dataclass
class TrainingConfig:
    run_name: str = 'tmp run'
    info: str = ""
    use_wandb: bool = False
    seed: int = 0
    devices: Optional[str] = 'None'

    theoretical_training: bool = False

    wrong_sent_ratio: Optional[str] = None
    overall_typo_ratio: Optional[str] = None
    typo_type_dist: Optional[str] = None

    n_epochs: int = 20

    train_filepath: str = "./data/train.txt"
    validate_filepath: str = "./data/dev.txt"
    labeled_filepath: str = "./data/test.txt"

    cache_root_dir: str = './cache_dir'
    train_cache_name: str = "train.tar"
    validate_cache_name: str = "dev.tar"
    labeled_cache_name: str = "test.tar"

    train_length: int = -1

    lr: float = 1e-4
    pretrained_model_lr: float = 2e-5
    accumulation_steps: int = 1
    warmup_epochs: int = 2

    batch_size: int = 128
    validate_batch_size: int = 256
    shuffle: bool = True

    save_model: bool = False
    checkpoint_root_dir: str = "./checkpoints/"
    checkpoint_name: str = "tmp"

    def __post_init__(self):
        self.devices = eval(self.devices)
        self.wrong_sent_ratio = eval(self.wrong_sent_ratio)
        self.overall_typo_ratio = eval(self.overall_typo_ratio)
        self.typo_type_dist = eval(self.typo_type_dist)

        Path(self.cache_root_dir).mkdir(exist_ok=True, parents=True)
        Path(self.checkpoint_root_dir).joinpath(self.checkpoint_name).mkdir(exist_ok=True, parents=True)

        logger.info(f'使用的 devices：{self.devices}.')
        if isinstance(self.devices, (List, Tuple)):
            logger.info(f'总共的 batch_size 为：{self.accumulation_steps * len(self.devices) * self.batch_size}.')
        else:
            logger.info(f'总共的 batch_size 为：{self.accumulation_steps * self.batch_size}.')


if __name__ == "__main__":
    parser = HfArgumentParser((ModelConfig, DataConfig, TrainingConfig))
    model_config, data_config, training_config = parser.parse_args_into_dataclasses()
    model_config.theoretical_training = training_config.theoretical_training
    data_config.theoretical_training = training_config.theoretical_training

    torch_seed_everything(seed=training_config.seed)

    if training_config.use_wandb:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            info_dict = dataclass_to_dict(model_config)
            info_dict.update(dataclass_to_dict(data_config))
            info_dict.update(dataclass_to_dict(training_config))
            wandb.init(
                project="pinyin",
                name=training_config.run_name,
                config=info_dict,
            )

        @Trainer.on(Event.on_before_backward())
        def _losses_logger(trainer, outputs):
            # 该 outputs 的结果来自于 ``train_step``；
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                _outputs = {}
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        _outputs[key] = value.item()
                    else:
                        _outputs[key] = value
                _outputs['global_forward_steps'] = trainer.global_forward_batches
                wandb.log(_outputs)

        @Trainer.on(Event.on_evaluate_end())
        def _wandb_logger(trainer, results):
            results = deepcopy(results)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                for key in list(results.keys()):
                    if key not in {"f1#acc#val_dataloader", "pre#acc#val_dataloader", "rec#acc#val_dataloader", "right_rec#acc#val_dataloader",
                    "wrong_rec#acc#val_dataloader", "sent_acc#acc#val_dataloader", "1_type_acc#acc#val_dataloader", "2_type_acc#acc#val_dataloader",
                    "3_type_acc#acc#val_dataloader", "4_type_acc#acc#val_dataloader", "5_type_acc#acc#val_dataloader", "6_type_acc#acc#val_dataloader"}:
                        results.pop(key)
                results['epoch'] = trainer.cur_epoch_idx
                results['global_forward_steps'] = trainer.global_forward_batches
                wandb.log(results)

    model = Model(model_config)
    def model_save_fn(folder):
        """
        这一函数需要定制，因为我们需要保存模型当中的一些特殊的属性设置；
        """
        folder = Path(folder)
        torch.save(dataclass_to_dict(model_config), folder.joinpath("model_config.tar"))
        states = {name: param.cpu().detach().clone() for name, param in model.state_dict().items()}
        torch.save(states, folder.joinpath("fastnlp_model.pkl.tar"))

    tokenizer = BertTokenizerFast.from_pretrained(model_config.pretrained_model)

    @cache_results(_cache_fp=make_cache_name(Path(training_config.cache_root_dir).joinpath(training_config.train_cache_name),
                                             training_config.train_length), _hash_param=False)
    def _load_train_dataset():
        data = load_simulated_data(
            filepath=training_config.train_filepath,
            wrong_sent_ratio=training_config.wrong_sent_ratio,
            overall_typo_ratio=training_config.overall_typo_ratio,
            typo_type_dist=training_config.typo_type_dist,
            length=training_config.train_length
        )
        return prepare_dataset(data, tokenizer, data_config, num_proc=128)
    train_dataset = _load_train_dataset()
    train_dataloader = prepare_torch_dataloader(train_dataset, batch_size=training_config.batch_size,
                                                shuffle=training_config.shuffle, collate_fn=collate_fn)

    @cache_results(_cache_fp=make_cache_name(Path(training_config.cache_root_dir).joinpath(training_config.validate_cache_name)), _hash_param=False)
    def _load_validate_dataset():
        # 验证集的错误率始终保持和 labeled data 一致；
        data = load_simulated_data(
            filepath=training_config.validate_filepath,
            wrong_sent_ratio=0.35,
            overall_typo_ratio=(0.72, 0.2, 0.08),
            typo_type_dist=(0.24, 0.16, 0.31, 0.1, 0.15, 0.04),
        )
        return prepare_dataset(data, tokenizer, data_config, num_proc=64)

    validate_dataset = _load_validate_dataset()
    validate_dataloader = prepare_torch_dataloader(validate_dataset, batch_size=training_config.validate_batch_size,
                                                   shuffle=False, collate_fn=collate_fn)

    metrics = {"acc": IntegrateAccuracy(predict_process_fn=predict_process_fn, target_process_fn=target_process_fn)}

    pretrained_model_optimizer = AdamW(model.pretrained_model.parameters(),
                                       lr=training_config.pretrained_model_lr, betas=(0.9, 0.99))
    decode_layer_optimizer = AdamW(model.decode_layer.parameters(), lr=training_config.lr, betas=(0.9, 0.99))
    used_optimizers = [decode_layer_optimizer, pretrained_model_optimizer]

    _warmup_fn = lambda epoch: epoch / training_config.warmup_epochs
    pretrained_warmup_scheduler = LambdaLR(pretrained_model_optimizer, lr_lambda=_warmup_fn)
    decode_warmup_scheduler = LambdaLR(decode_layer_optimizer, lr_lambda=_warmup_fn)
    pretrained_cos_scheduler = CosineAnnealingWarmRestarts(pretrained_model_optimizer, T_0=10 + training_config.warmup_epochs, T_mult=2)
    decode_cos_scheduler = CosineAnnealingWarmRestarts(decode_layer_optimizer, T_0=10 + training_config.warmup_epochs, T_mult=2)

    callbacks = [
        LRSchedCallback(pretrained_warmup_scheduler, 0, step_on='epoch', name="pretrained_warmup_scheduler"),
        LRSchedCallback(decode_warmup_scheduler, 0, step_on='epoch', name="decode_warmup_scheduler"),
        LRSchedCallback(pretrained_cos_scheduler, training_config.warmup_epochs, step_on='batch',
                        name="pretrained_cos_scheduler"),
        LRSchedCallback(decode_cos_scheduler, training_config.warmup_epochs, step_on='batch',
                        name="decode_cos_scheduler"),

        EmpetyCudaCache()
    ]

    if training_config.save_model:
        callbacks.append(
            CheckpointCallback(
                monitor="f1#acc#val_dataloader",
                folder=Path(training_config.checkpoint_root_dir).joinpath(training_config.checkpoint_name),
                topk=1,
                last=True,
                save_object="model",
                model_save_fn=model_save_fn
            )
        )

    if model_config.boundary_predict:
        boundary_layer_optimizer = AdamW(model.boundary_layer.parameters(), lr=training_config.lr, betas=(0.9, 0.99))
        used_optimizers.append(boundary_layer_optimizer)
        boundary_warmup_scheduler = LambdaLR(boundary_layer_optimizer, lr_lambda=_warmup_fn)
        boundary_cos_scheduler = CosineAnnealingWarmRestarts(boundary_layer_optimizer,
                                                             T_0=10 + training_config.warmup_epochs, T_mult=2)
        callbacks.extend([
            LRSchedCallback(boundary_warmup_scheduler, 0, step_on='epoch', name="boundary_warmup_scheduler"),
            LRSchedCallback(boundary_cos_scheduler, training_config.warmup_epochs, step_on='batch',
                            name="boundary_cos_scheduler"),
            MoreEvaluateCallback(dataloaders={"val_dataloader": validate_dataloader},
                                 metrics={"boundary_acc": CharacterBoundaryAccuracyPinyin()},
                                 evaluate_fn="evaluate_boundary_step",
                                 evaluate_every=-4)
        ])

    if model_config.character_boundary_predict:
        character_boundary_layer_optimizer = AdamW(model.character_boundary_layer.parameters(), lr=training_config.lr, betas=(0.9, 0.99))
        used_optimizers.append(character_boundary_layer_optimizer)
        character_boundary_warmup_scheduler = LambdaLR(character_boundary_layer_optimizer, lr_lambda=_warmup_fn)
        character_boundary_cos_scheduler = CosineAnnealingWarmRestarts(character_boundary_layer_optimizer,
                                                                       T_0=10 + training_config.warmup_epochs, T_mult=2)
        callbacks.extend([
            LRSchedCallback(character_boundary_warmup_scheduler, 0, step_on='epoch', name="character_boundary_warmup_scheduler"),
            LRSchedCallback(character_boundary_cos_scheduler, training_config.warmup_epochs, step_on='batch',
                            name="character_boundary_cos_scheduler"),
            MoreEvaluateCallback(dataloaders={"val_dataloader": validate_dataloader},
                                 metrics={"character_boundary_acc": CharacterBoundaryAccuracyPinyin()},
                                 evaluate_fn="evaluate_character_boundary_step",
                                 evaluate_every=-4)
        ])

    if model_config.use_lstm:
        lstm_optimizer = AdamW(model.lstm.parameters(), lr=training_config.lr, betas=(0.9, 0.99))
        used_optimizers.append(lstm_optimizer)
        lstm_warmup_scheduler = LambdaLR(lstm_optimizer, lr_lambda=_warmup_fn)
        lstm_cos_scheduler = CosineAnnealingWarmRestarts(lstm_optimizer,
                                                         T_0=10 + training_config.warmup_epochs, T_mult=2)
        callbacks.extend([
            LRSchedCallback(lstm_warmup_scheduler, 0, step_on='epoch', name="lstm_cos_scheduler"),
            LRSchedCallback(lstm_cos_scheduler, training_config.warmup_epochs, step_on='batch',
                            name="lstm_cos_scheduler"),
        ])

    @Trainer.on(Event.on_before_optimizers_step())
    def _grad_clip(trainer, optimizers):
        nn.utils.clip_grad_norm_(trainer.model.parameters(), 0.01)
        for params in trainer.model.parameters():
            if params.grad is not None:
                torch.nan_to_num_(params.grad, nan=0, posinf=0, neginf=0)

    trainer = Trainer(
        model=model,
        driver="torch",
        device=training_config.devices,
        optimizers=used_optimizers,
        train_dataloader=train_dataloader,
        evaluate_dataloaders={"val_dataloader": validate_dataloader},
        metrics=metrics,
        output_mapping=None,
        model_wo_auto_param_call=True,
        output_from_new_proc="all",
        n_epochs=training_config.n_epochs,
        callbacks=callbacks,
        fp16=False,
        accumulation_steps=training_config.accumulation_steps,
        evaluate_every=-2,
        torch_kwargs={"ddp_kwargs": {"find_unused_parameters": True}}
    )

    trainer.run()

    logger.info('训练完毕，开始在标注集上测试；')

    @cache_results(_cache_fp=make_cache_name(Path(training_config.cache_root_dir).joinpath(training_config.labeled_cache_name)), _hash_param=False)
    def _load_test_dataset():
        data = load_labeled_data(filepath=training_config.labeled_filepath)
        return prepare_dataset(data, tokenizer, data_config, num_proc=64)

    test_dataset = _load_test_dataset()
    test_dataloader = prepare_torch_dataloader(test_dataset, batch_size=training_config.validate_batch_size,
                                               shuffle=False, collate_fn=collate_fn)
    evaluator = Evaluator(
        model=None,  # 使用 trainer.driver.model；
        dataloaders={'test_dataloader': test_dataloader},
        metrics=metrics,
        driver=trainer.driver,
    )
    evaluator.run()







