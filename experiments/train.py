# Created by Hansi at 10/6/2021
import logging
import os

from algo.models.classification_model import ClassificationModel
from algo.models.ner_model import NERModel
from algo.util.file_util import delete_create_folder
from experiments import classifier_config
from experiments import ner_config
from farm.conversion.transformers import Converter
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.tokenization import Tokenizer
from farm.utils import set_all_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_classifier(data_dir, config):
    delete_create_folder(classifier_config.OUTPUT_DIRECTORY)
    base_model_dir = config['model_dir']
    base_train_progress_file_name = config['train_progress_file']
    base_file_name_splits = os.path.splitext(base_train_progress_file_name)

    for i in config["fold_ids"]:
        # update model dir for the fold
        config['model_dir'] = f"{base_model_dir}_{i}"
        # train_progress_file_name = os.path.basename(config['train_progress_file'])
        # file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(os.path.dirname(config['train_progress_file']),
                                                     f"{base_file_name_splits[0]}_{i}{base_file_name_splits[1]}")

        set_all_seeds(seed=int(config['manual_seed'] * (i + 1)))
        logger.info(f"Set seed to {int(config['manual_seed'] * (i + 1))}.")

        if classifier_config.MODEL_NAME is not None:
            model = ClassificationModel(classifier_config.MODEL_NAME, args=config)
        else:
            if classifier_config.MODEL_DIRECTORY is None:
                raise ValueError(f"Neither model name nor directory is defined!")
            else:
                model = ClassificationModel(os.path.join(classifier_config.MODEL_DIRECTORY, f"model_{i}"), args=config)
        logger.info(f"Training model for fold {i}...")
        model.train_model(data_dir)


def train_ner(data_dir, config):
    delete_create_folder(ner_config.OUTPUT_DIRECTORY)
    base_model_dir = config['model_dir']
    base_train_progress_file_name = config['train_progress_file']
    base_file_name_splits = os.path.splitext(base_train_progress_file_name)

    for i in config["fold_ids"]:
        # update model dir for the fold
        config['model_dir'] = f"{base_model_dir}_{i}"
        # train_progress_file_name = os.path.basename(config['train_progress_file'])
        # file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(os.path.dirname(config['train_progress_file']),
                                                     f"{base_file_name_splits[0]}_{i}{base_file_name_splits[1]}")

        set_all_seeds(seed=int(config['manual_seed'] * (i + 1)))
        logger.info(f"Set seed to {int(config['manual_seed'] * (i + 1))}.")

        if ner_config.MODEL_NAME is not None:
            model = NERModel(ner_config.MODEL_NAME, args=config)
        else:
            if ner_config.MODEL_DIRECTORY is None:
                raise ValueError(f"Neither model name nor directory is defined!")
            else:
                model = NERModel(os.path.join(ner_config.MODEL_DIRECTORY, f"model_{i}"), args=config)
        logger.info(f"Training model for fold {i}...")
        model.train_model(data_dir)


# def convert():
#     path = os.path.join(classifier_config.MODEL_DIRECTORY, f"model_0")
#     model = Converter.convert_from_transformers(path, device="cpu")
#     # model.save(os.path.join(classifier_config.MODEL_DIRECTORY, f"model_1"))
#
#     tokenizer = Tokenizer.load(pretrained_model_name_or_path=path, do_lower_case=True)
#     processor = TextClassificationProcessor(tokenizer=tokenizer,
#                                             max_seq_len=128,
#                                             data_dir="data",
#                                             label_list=[],
#                                             label_column_name="label",
#                                             metric="acc",
#                                             quote_char='"',
#                                             )
#
#     model.connect_heads_with_processor(processor.tasks, require_labels=True)
#
#     save_dir = os.path.join(classifier_config.MODEL_DIRECTORY, f"model_2")
#     model.save(save_dir)
#     processor.save(save_dir)


if __name__ == '__main__':
    # train classifier
    # data_dir = os.path.join(classifier_config.DATA_DIRECTORY, 'subtask2-sentence/filtered/farm_format')
    # train_classifier(data_dir, classifier_config.config)

    # train ner
    # data_dir = os.path.join(ner_config.DATA_DIRECTORY, 'subtask4-token/filtered/farm_format')
    data_dir = os.path.join(ner_config.DATA_DIRECTORY, 'subtask4-token/filtered/farm_format/split_binary')
    train_ner(data_dir, ner_config.config)
