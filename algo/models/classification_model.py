# Created by Hansi at 9/23/2021
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import EarlyStopping, Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from algo.models.config.model_args import ClassificationModelArgs

logger = logging.getLogger(__name__)


class ClassificationModel:
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationModelArgs):
            self.args = args

        # set_all_seeds(seed=self.args.manual_seed)
        self.device, self.n_gpu = initialize_device_settings(use_cuda=self.args.use_cuda, use_amp=self.args.use_amp)

    def train_model(self, data_dir):
        # create tokenizer
        tokenizer = Tokenizer.load(pretrained_model_name_or_path=self.model_name,
                                   do_lower_case=self.args.do_lower_case)

        # create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        processor = TextClassificationProcessor(tokenizer=tokenizer,
                                                max_seq_len=self.args.max_seq_len,
                                                data_dir=Path(data_dir),
                                                label_list=self.args.label_list,
                                                metric=self.args.metric,
                                                train_filename=self.args.train_filename,
                                                dev_filename=self.args.dev_filename,
                                                test_filename=self.args.test_filename,
                                                dev_split=self.args.dev_split,
                                                dev_stratification=self.args.dev_stratification,
                                                delimiter=self.args.delimiter,
                                                label_column_name=self.args.label_column_name,
                                                text_column_name=self.args.text_column_name,
                                                multilabel=self.args.multilabel
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
        prediction_head = TextClassificationHead(
            class_weights=data_silo.calculate_class_weights(task_name=self.args.task_name),
            num_labels=len(self.args.label_list))
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

    def predict(self, texts, inference_batch_size):
        """
        predict labels for given samples

        :param texts: list of dict {'text': "sample text"}
        :param inference_batch_size: int
        :return: list, list
            predictions- list of class labels
            raw predictions- list of dict {'start': None, 'end': None, 'context':"sample text", 'label': 'predicted label', 'probability': 0.9404173}
        """
        model = Inferencer.load(self.args.model_dir, batch_size=inference_batch_size)
        result = model.inference_from_dicts(dicts=texts)
        model.close_multiprocessing_pool()

        raw_predictions = []
        for idx, chunk_res in enumerate(result):
            raw_predictions += chunk_res["predictions"]

        predictions = [x['label'] for x in raw_predictions]
        return predictions, raw_predictions

    def _load_model_args(self, input_dir):
        args = ClassificationModelArgs()
        args.load(input_dir)
        return args
