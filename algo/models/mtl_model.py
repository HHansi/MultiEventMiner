# Created by Hansi at 11/15/2021
import logging
from pathlib import Path
from typing import List

import torch
from sklearn.metrics import f1_score

from algo.models.config.model_args import MTLModelArgs
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.mtl_processor import MTLProcessor
from farm.evaluation.metrics import register_metrics
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead, TokenClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import EarlyStopping, Trainer
from farm.utils import initialize_device_settings

logger = logging.getLogger(__name__)


def token_macro_f1(y_true, y_pred):
    f1_scores = []
    for t, p in zip(y_true, y_pred):
        f1_scores.append(f1_score(t, p, average="macro"))
    return {"F1 macro score": sum(f1_scores) / len(f1_scores), "Total": len(f1_scores)}


def loss_function(individual_losses: List[torch.Tensor], global_step=None, batch=None, alpha=1, beta=1):
    loss = (alpha * torch.sum(individual_losses[0]) + beta * torch.sum(individual_losses[1])) / (alpha + beta)
    return loss


class MTLModel:
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, MTLModelArgs):
            self.args = args

        # set_all_seeds(seed=self.args.manual_seed)
        self.device, self.n_gpu = initialize_device_settings(use_cuda=self.args.use_cuda, use_amp=self.args.use_amp)

        if (isinstance(self.args.token_metric,
                       list) and "token_f1" in self.args.token_metric) or self.args.token_metric == "token_f1":
            register_metrics("token_f1", token_macro_f1)

    def train_model(self, data_dir):
        # create tokenizer
        tokenizer = Tokenizer.load(pretrained_model_name_or_path=self.model_name,
                                   do_lower_case=self.args.do_lower_case)

        ner_labels = ["[PAD]", "X"] + self.args.token_label_list

        # create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        processor = MTLProcessor(tokenizer=tokenizer,
                                 max_seq_len=self.args.max_seq_len,
                                 data_dir=Path(data_dir),
                                 text_label_list=self.args.text_label_list,
                                 token_label_list=ner_labels,
                                 text_metric=self.args.text_metric,
                                 token_metric=self.args.token_metric,
                                 train_filename=self.args.train_filename,
                                 dev_filename=self.args.dev_filename,
                                 test_filename=self.args.test_filename,
                                 dev_split=self.args.dev_split,
                                 delimiter=self.args.delimiter,
                                 text_column_name=self.args.text_column_name,
                                 text_label_column_name=self.args.text_label_column_name,
                                 token_label_column_name=self.args.token_label_column_name
                                 )

        # create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
        #    few descriptive statistics of our datasets
        data_silo = DataSilo(
            processor=processor,
            max_processes=self.args.max_processes,
            batch_size=self.args.train_batch_size,
            eval_batch_size=self.args.eval_batch_size
        )

        # create an AdaptiveModel
        language_model = LanguageModel.load(self.model_name)
        document_level_task_head = TextClassificationHead(
            num_labels=len(self.args.text_label_list), task_name="document_level_task"
        )
        token_level_task_head = TokenClassificationHead(
            num_labels=len(ner_labels), task_name="token_level_task"
        )
        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[document_level_task_head, token_level_task_head],
            embeds_dropout_prob=self.args.embeds_dropout_prob,
            lm_output_types=self.args.lm_output_types,
            device=self.device,
            loss_aggregation_fn=loss_function)

        # create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=self.args.learning_rate,
            device=self.device,
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=self.args.n_epochs,
            use_amp=self.args.use_amp)

        early_stopping = None
        if self.args.use_early_stopping:
            early_stopping = EarlyStopping(
                metric=self.args.early_stopping_metric, mode=self.args.early_stopping_mode,
                save_dir=self.args.model_dir, patience=self.args.early_stopping_patience
            )

        # feed everything to the Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=self.args.n_epochs,
            n_gpu=self.n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=self.args.evaluate_every,
            device=self.device,
            use_amp=self.args.use_amp,
            grad_acc_steps=self.args.gradient_accumulation_steps,
            early_stopping=early_stopping,
            max_grad_norm=self.args.max_grad_norm,
            train_progress_file=self.args.train_progress_file
        )
        trainer.train()

        if not early_stopping:
            model.save(self.args.model_dir)
            processor.save(self.args.model_dir)

        del model
        del processor
        del optimizer
        del data_silo
        del trainer

    def _load_model_args(self, input_dir):
        args = MTLModelArgs()
        args.load(input_dir)
        return args
