# Created by Hansi at 10/4/2021
import os
from multiprocessing import cpu_count

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

# TEMP_DIRECTORY = "temp"
SUBMISSION_FILE = os.path.join(OUTPUT_DIRECTORY, 'submission.txt')

MODEL_NAME = "bert-base-cased"
LANGUAGES = ["en"]

config = {
    'manual_seed': SEED,
    'model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),
    'train_progress_file': os.path.join(OUTPUT_DIRECTORY, "training_progress_scores.csv"),

    'do_lower_case': False,
    'max_seq_len': 128,
    'n_epochs': 4,
    'train_batch_size': 2,  # 8
    'eval_batch_size': 2,  # 8
    'inference_batch_size': 2,  # 8
    'evaluate_every': 1,  # 4
    'learning_rate': 1e-5,  # 3e-5
    'metric': "seq_f1",
    'use_early_stopping': True,
    'early_stopping_metric': "loss",
    'early_stopping_mode': "min",  # "min" or "max"
    'early_stopping_patience': 10,

    'label_list': ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH", "I-OTH"],
    'train_filename': "train.txt",
    'dev_filename': "valid.txt",
    'dev_split': 0.1,
    'delimiter': " ",
    'max_processes': 128,  # cpu_count() - 2 if cpu_count() > 2 else 1,  # 128 is default
    'use_amp': None,

    'n_fold': 1
}
