# Created by Hansi at 10/4/2021
import logging
from pathlib import Path

from algo.models.common.eval import token_macro_f1
from algo.models.common.ner_util import to_iob, to_binary
from algo.models.config.model_args import NERModelArgs
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NERProcessor
from farm.evaluation.metrics import register_metrics
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import EarlyStopping, Trainer
from farm.utils import initialize_device_settings

logger = logging.getLogger(__name__)


class NERModel:
    def __init__(self, model_name, args, mode='train'):
        self.model_name = model_name
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, NERModelArgs):
            self.args = args

        # set_all_seeds(seed=self.args.manual_seed)
        self.device, self.n_gpu = initialize_device_settings(use_cuda=self.args.use_cuda, use_amp=self.args.use_amp)

        if (isinstance(self.args.metric, list) and "token_f1" in self.args.metric) or self.args.metric == "token_f1":
            register_metrics("token_f1", token_macro_f1)

        if mode == 'inference':
            self.model = Inferencer.load(self.args.model_dir, batch_size=self.args.inference_batch_size,
                                         max_seq_len=self.args.max_seq_len, gpu=self.args.gpu,
                                         num_processes=self.args.num_processes)

    def train_model(self, data_dir):
        """
        train sequence labelling model

        :param data_dir: path to directory which holds training data
        :return:
        """
        # create tokenizer
        tokenizer = Tokenizer.load(pretrained_model_name_or_path=self.model_name,
                                   do_lower_case=self.args.do_lower_case)

        ner_labels = ["[PAD]", "X"] + self.args.label_list

        # create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        processor = NERProcessor(tokenizer=tokenizer,
                                 max_seq_len=self.args.max_seq_len,
                                 data_dir=Path(data_dir),
                                 label_list=ner_labels,
                                 metric=self.args.metric,
                                 train_filename=self.args.train_filename,
                                 dev_filename=self.args.dev_filename,
                                 test_filename=self.args.test_filename,
                                 dev_split=self.args.dev_split,
                                 delimiter=self.args.delimiter,
                                 # quote_char='"'  # Quote chars are used so that text can include the tsv delimiter symbol (i.e. \t) without ruining the tsv format
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
        prediction_head = TokenClassificationHead(
            task_name=self.args.task_name,
            num_labels=len(ner_labels))
        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=self.args.embeds_dropout_prob,
            lm_output_types=self.args.lm_output_types,
            device=self.device)

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

    def predict(self, texts):
        """
        predict labels for given samples

        :param texts: list of dict {'text': "sample text"}
        :param inference_batch_size: int
        :return: list, list
            predictions- list of class labels
            raw predictions- list of dict {'start': None, 'end': None, 'context':"sample text", 'label': 'predicted label', 'probability': 0.9404173}
        """
        result = self.model.inference_from_dicts(dicts=texts)
        self.model.close_multiprocessing_pool()

        raw_predictions = []
        for idx, chunk_res in enumerate(result):
            raw_predictions += chunk_res["predictions"]

        if self.args.label_format == "iob":
            predictions = to_iob(texts, raw_predictions)
        elif self.args.label_format == "binary":
            predictions = to_binary(texts, raw_predictions)
        else:
            raise KeyError(f"Label output format is not defined!")
        return predictions, raw_predictions

    def _load_model_args(self, input_dir):
        args = NERModelArgs()
        args.load(input_dir)
        return args
