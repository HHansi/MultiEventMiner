# Created by Hansi at 10/25/2021
import os

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

MODEL_NAME = "bert-base-cased"

config = {
    'manual_seed': SEED,
    'model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),  # folder to save the trained model
    'train_progress_file': os.path.join(OUTPUT_DIRECTORY, "training_progress_scores.csv"),

    'do_lower_case': False,
    'max_seq_len': 128,
    'n_epochs': 2,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'inference_batch_size': 16,
    'evaluate_every': 1000,
    'learning_rate': 1e-5,
    'use_early_stopping': True,
    'early_stopping_metric': "loss",
    'early_stopping_mode': "min",  # "min" or "max"
    'early_stopping_patience': 10,

    'train_filename': "en-lm.txt",
    'dev_split': 0.1,
    'max_processes': 128,  # 128 is default
    'use_amp': None,

    'fold_ids': [0],  # list of ids for folds
}

