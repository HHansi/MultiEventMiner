# Created by Hansi at 11/15/2021

import os
from multiprocessing import cpu_count

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'predictions')
SUBMISSION_FILE = os.path.join(PREDICTION_DIRECTORY, 'submission.txt')

# Either one (model name or directory) should be provided. Other should be set to None.
# If both provided, only the model name will be considered.
MODEL_NAME = "bert-large-cased"
MODEL_DIRECTORY = ""  # Use if multiple models need to be referred during training (model name = model_<fold_id>).
LANGUAGES = ["en"]

config = {
    'manual_seed': SEED,
    'model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),  # folder to save the trained model
    'train_progress_file': os.path.join(OUTPUT_DIRECTORY, "training_progress_scores.csv"),

    'do_lower_case': False,
    'max_seq_len': 128,
    'n_epochs': 3,  # 4
    'train_batch_size': 8,  # 2
    'eval_batch_size': 8,  # 2
    'inference_batch_size': 16,  # 2
    'evaluate_every': 80,  # 1
    'learning_rate': 1e-5,

    'use_early_stopping': True,
    'early_stopping_metric': "loss",
    'early_stopping_mode': "min",  # "min" or "max"
    'early_stopping_patience': 10,
    'label_format': "binary",   # "iob", "binary"

    'text_label_list': ["0", "1"],
    'token_label_list': ["0", "1"],
    # 'token_label_list': ["O", "B-trigger", "I-trigger", "B-target", "I-target", "B-place", "I-place", "B-etime", "I-etime",
    #               "B-fname", "I-fname", "B-participant", "I-participant", "B-organizer", "I-organizer"],

    'train_filename': "en-train-binary.csv",
    'dev_filename': None,
    'dev_split': 0.1,
    'delimiter': ",",
    'text_column_name': "text_tokens",
    'text_label_column_name': "text_label",
    'token_label_column_name': "token_label_binary",

    'text_metric': ["f1_macro", "acc"],
    'token_metric': "token_f1",  # "seq_f1"
            
    'max_processes': 128,  # 128 is default
    'use_amp': None,
    'cude_device': "1",  # or None

    'n_fold': 1,  # number of models to train for ensemble
    'fold_ids': [0],  # list of ids for folds

    # for inferencer
    'gpu': True,
    'num_processes': cpu_count() - 2 if cpu_count() > 2 else 1
}

