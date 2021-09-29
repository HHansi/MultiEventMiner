# Created by Hansi at 9/21/2021
import os

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

# TEMP_DIRECTORY = "temp"
SUBMISSION_FILE = os.path.join(OUTPUT_DIRECTORY, 'submission.json')

MODEL_NAME = "bert-base-cased"
LANGUAGES = ["en"]

config = {
    'manual_seed': SEED,
    'model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),
    'train_progress_file': os.path.join(OUTPUT_DIRECTORY, "training_progress_scores.csv"),

    'do_lower_case': False,
    'max_seq_len': 128,
    'n_epochs': 2,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'inference_batch_size': 4,
    'evaluate_every': 4,
    'learning_rate': 3e-5,
    'metric': ["f1_macro", "acc"],
    'use_early_stopping': True,
    'early_stopping_metric': "loss",
    'early_stopping_mode': "min",  # "min" or "max"
    'early_stopping_patience': 5,

    'label_list': ["0", "1"],
    'train_filename': "en-train2.tsv",
    'dev_split': 0.1,
    'dev_stratification': True,
    'delimiter': "\t",
    'label_column_name': "label",  # for classification
    'text_column_name': "text",
    'max_processes': 1,  # 128 is default
    'use_amp': None,

    'n_fold': 1
}


