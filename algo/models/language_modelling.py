# Created by Hansi at 10/24/2021
from pathlib import Path

from algo.models.config.model_args import LMArgs
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import BertLMHead, NextSentenceHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer, EarlyStopping
from farm.utils import initialize_device_settings


class LanguageModellingModel:
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, LMArgs):
            self.args = args

        # set_all_seeds(seed=self.args.manual_seed)
        self.device, self.n_gpu = initialize_device_settings(use_cuda=self.args.use_cuda, use_amp=self.args.use_amp)

    def train_model(self, data_dir):
        if "bert" not in self.model_name.lower():
            raise ValueError("LM is only supported for BERT models!")

        # create tokenizer
        tokenizer = Tokenizer.load(pretrained_model_name_or_path=self.model_name, do_lower_case=self.args.do_lower_case
                                   )

        # create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        processor = BertStyleLMProcessor(
            tokenizer=tokenizer,
            max_seq_len=self.args.max_seq_len,
            data_dir=Path(data_dir),
            train_filename=self.args.train_filename,
            dev_filename=self.args.dev_filename,
            test_filename=self.args.test_filename,
            dev_split=self.args.dev_split,
            next_sent_pred=self.args.next_sent_pred
        )

        # create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
        data_silo = DataSilo(
            processor=processor,
            batch_size=self.args.train_batch_size,
            max_processes=self.args.max_processes,
            max_multiprocessing_chunksize=20)

        # create an AdaptiveModel
        # a) which consists of a pretrained language model as a basis
        language_model = LanguageModel.load(self.model_name)
        # b) and *two* prediction heads on top that are suited for our task => Language Model finetuning
        lm_prediction_head = BertLMHead.load(self.model_name)
        next_sentence_head = NextSentenceHead.load(self.model_name)

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[lm_prediction_head, next_sentence_head],
            embeds_dropout_prob=self.args.embeds_dropout_prob,
            lm_output_types=self.args.lm_output_types,
            device=self.device,
        )

        # create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=self.args.learning_rate,
            device=self.device,
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=self.args.n_epochs,
            use_amp=self.args.use_amp
        )

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
            train_progress_file=self.args.train_progress_file,
            eval_report=False
        )

        trainer.train()

        if not early_stopping:
            model.save(self.args.model_dir)
            processor.save(self.args.model_dir)

    def _load_model_args(self, input_dir):
        args = LMArgs()
        args.load(input_dir)
        return args
