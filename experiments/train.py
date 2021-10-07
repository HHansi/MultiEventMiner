# Created by Hansi at 10/6/2021
import logging
import os

from algo.models.classification_model import ClassificationModel
from algo.models.ner_model import NERModel
from algo.util.file_util import delete_create_folder
from experiments import classifier_config
from experiments import ner_config
from farm.utils import set_all_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_classifier(data_dir, config):
    delete_create_folder(classifier_config.OUTPUT_DIRECTORY)

    for i in config["fold_ids"]:
        # update model dir for the fold
        config['model_dir'] = f"{config['model_dir']}_{i}"
        train_progress_file_name = os.path.basename(config['train_progress_file'])
        file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(
            os.path.dirname(config['train_progress_file']),
            f"{file_name_splits[0]}_{i}{file_name_splits[1]}")

        set_all_seeds(seed=int(config['manual_seed'] * (i + 1)))
        logger.info(f"Set seed to {int(config['manual_seed'] * (i + 1))}.")

        model = ClassificationModel(classifier_config.MODEL_NAME, args=config)
        logger.info(f"Training model for fold {i}...")
        model.train_model(data_dir)


def train_ner(data_dir, config):
    delete_create_folder(ner_config.OUTPUT_DIRECTORY)

    for i in config["fold_ids"]:
        # update model dir for the fold
        config['model_dir'] = f"{config['model_dir']}_{i}"
        train_progress_file_name = os.path.basename(config['train_progress_file'])
        file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(os.path.dirname(config['train_progress_file']),
                                                     f"{file_name_splits[0]}_{i}{file_name_splits[1]}")

        set_all_seeds(seed=int(config['manual_seed'] * (i + 1)))
        logger.info(f"Set seed to {int(config['manual_seed'] * (i + 1))}.")

        model = NERModel(ner_config.MODEL_NAME, args=config)
        logger.info(f"Training model for fold {i}...")
        model.train_model(data_dir)


if __name__ == '__main__':
    # train classifier
    data_dir = os.path.join(classifier_config.DATA_DIRECTORY, 'subtask2-sentence/filtered/farm_format')
    train_classifier(data_dir, classifier_config.config)

    # train ner
    # data_dir = os.path.join(ner_config.DATA_DIRECTORY, 'subtask4-token/filtered/farm_format')
    # train_classifier(data_dir, ner_config.config)

