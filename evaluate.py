

import argparse
from pathlib import Path
from typing import Dict

import torch
from transformers import BertTokenizerFast

from fastNLP import logger, cache_results, prepare_torch_dataloader, Evaluator

from models.model import Model, ModelConfig
from models.metric import IntegrateAccuracy, BoundaryAccuracyPinyin, CharacterBoundaryAccuracyPinyin
from models.utils import predict_process_fn, target_process_fn, EmpetyCudaCache, LRSchedCallback, make_cache_name, dataclass_to_dict
from models.data_preprocess import DataConfig, load_simulated_data, load_labeled_data, prepare_dataset, collate_fn


class DummyDataclass:

    def __init__(self, data):
        self.data = data

    def __getattr__(self, item):
        return self.data[item]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="保存的模型参数的文件夹。")
    parser.add_argument("--data_path", help="用于评测的实际标注的文件位置。")
    parser.add_argument("--data_cache_path", help="如果该值被设置，那么我们会默认加载该值指定的缓存数据集。", default="./tmp")
    parser.add_argument("--device", help="使用哪张 gpu。", default=0)

    args = parser.parse_args()

    device = [int(w) for w in args.device.split()]
    if len(device) == 1:
        device = device[0]

    model_path = Path(args.model_path)
    training_config = torch.load(model_path.joinpath("training_config.tar"))
    model_config = torch.load(model_path.joinpath("model_config.tar"))
    data_config = torch.load(model_path.joinpath("data_config.tar"))

    model_config['theoretical_training'] = training_config['theoretical_training']
    data_config['theoretical_training'] = training_config['theoretical_training']

    training_config = DummyDataclass(training_config)
    model_config = DummyDataclass(model_config)
    data_config = DummyDataclass(data_config)

    tokenizer = BertTokenizerFast.from_pretrained(model_config.pretrained_model)

    model = Model(model_config)
    state_dict = torch.load(model_path.joinpath("fastnlp_model.pkl.tar"))
    model.load_state_dict(state_dict)

    @cache_results(_cache_fp=args.data_cache_path, _hash_param=False)
    def _load_test_dataset():
        data = load_labeled_data(filepath=args.data_path)
        return prepare_dataset(data, tokenizer, data_config, num_proc=64)


    test_dataset = _load_test_dataset()
    test_dataloader = prepare_torch_dataloader(test_dataset, batch_size=256,
                                               shuffle=False, collate_fn=collate_fn)

    metrics = {"acc": IntegrateAccuracy(predict_process_fn=predict_process_fn, target_process_fn=target_process_fn)}

    evaluator = Evaluator(
        driver="torch",
        device=device,
        model=model,  # 使用 trainer.driver.model；
        dataloaders={'test_dataloader': test_dataloader},
        metrics=metrics,
        model_wo_auto_param_call=True,
    )
    evaluator.run(num_eval_batch_per_dl=-1)























