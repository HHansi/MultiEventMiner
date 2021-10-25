# Created by Hansi at 10/24/2021
import logging
import os

from algo.models.language_modelling import LanguageModellingModel
from algo.util.file_util import delete_create_folder
from experiments.lm_config import OUTPUT_DIRECTORY, DATA_DIRECTORY, config, MODEL_NAME
from farm.utils import set_all_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    delete_create_folder(OUTPUT_DIRECTORY)
    data_dir = os.path.join(DATA_DIRECTORY, 'subtask1-doc/farm_format')

    for i in config["fold_ids"]:
        # update model dir for the fold
        config['model_dir'] = f"{config['model_dir']}_{i}"
        train_progress_file_name = os.path.basename(config['train_progress_file'])
        file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(os.path.dirname(config['train_progress_file']),
                                                     f"{file_name_splits[0]}_{i}{file_name_splits[1]}")
        set_all_seeds(seed=int(config['manual_seed'] * (i + 1)))
        logger.info(f"Set seed to {int(config['manual_seed'] * (i + 1))}")

        model = LanguageModellingModel(MODEL_NAME, args=config)
        logger.info(f"Training model for fold {i}...")
        model.train_model(data_dir)

