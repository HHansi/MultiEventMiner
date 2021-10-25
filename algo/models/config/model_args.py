# Created by Hansi at 9/23/2021
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class ModelArgs:
    # device
    use_cuda: bool = True
    use_amp: str = None
    max_processes: int = 1

    do_lower_case: bool = False
    max_seq_len: int = 128
    n_epochs: int = 1
    train_batch_size: int = 8
    embeds_dropout_prob: float = 0.1  # The probability that a value in the embeddings returned by the language model will be zeroed.
    learning_rate: float = 3e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # eval
    eval_batch_size: int = 8
    evaluate_every: int = 1
    use_early_stopping: bool = True
    early_stopping_metric: str = "loss"
    early_stopping_mode: str = "min"  # "min" or "max"
    early_stopping_patience: int = 5
    model_dir: str = "outputs/model"
    train_progress_file: str = "outputs/training_progress_scores.csv"

    # inferencing
    inference_batch_size: int = 8
    gpu: bool = False
    num_processes: int = 1

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {key: value for key, value in asdict(self).items() if key not in self.not_saved_args}
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            if args_dict["tokenizer_type"] is not None and not isinstance(args_dict["tokenizer_type"], str):
                args_dict["tokenizer_type"] = type(args_dict["tokenizer_type"]).__name__
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class ClassificationModelArgs(ModelArgs):
    task_name: str = "text_classification"
    lm_output_types = ["per_sequence"]

    label_list: list = None  # list of labels to predict (strings)
    train_filename: str = "train.tsv"  # for classification
    dev_filename: str = None
    dev_split: float = 0.1
    dev_stratification = False  # if true, create a class-stratified split for the dev set. Stratified split preserves
    # the same proportions of examples in each class as observed in the original dataset
    test_filename: str = None
    multilabel: bool = False  # set to True for multilabel classification
    # data format
    delimiter = "\t"
    label_column_name = "label"  # for classification
    text_column_name = "text"  # for classification
    # eval
    metric = ["f1_macro", "acc"]  # for classification


@dataclass
class NERModelArgs(ModelArgs):
    task_name: str = "ner"
    lm_output_types = ["per_token"]

    label_list: list = None  # I and B and O tags
    train_filename: str = "train.txt"
    dev_filename: str = None
    dev_split: float = 0.1
    test_filename: str = None
    # data format
    delimiter = "\t"
    # eval
    metric = "seq_f1"


@dataclass
class LMArgs(ModelArgs):
    lm_output_types = ["per_token", "per_sequence"]

    train_filename: str = "train.tsv"
    dev_filename: str = None
    dev_split: float = 0.1
    test_filename: str = None

