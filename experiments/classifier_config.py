# Created by Hansi at 9/21/2021
import os
from multiprocessing import cpu_count

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

# TEMP_DIRECTORY = "temp"
PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'predictions')
SUBMISSION_FILE = os.path.join(PREDICTION_DIRECTORY, 'submission.json')

# Either one (model name or directory) should be provided. Other should be set to None.
# If both provided, only the model name will be considered.
MODEL_NAME = "neuralmind/bert-large-portuguese-cased"  # "xlm-roberta-large"
MODEL_DIRECTORY = None  # "/experiments/tranasinghe/MultiEventMiner/trained_models2/token/xlm-r-large-en"  # Use if multiple models need to be referred during training (model name = model_<fold_id>).
LANGUAGES = ["pr"]

config = {
    'manual_seed': SEED,
    'model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),  # folder to save the trained model
    'train_progress_file': os.path.join(OUTPUT_DIRECTORY, "training_progress_scores.csv"),

    'do_lower_case': False,
    'max_seq_len': 128,
    'n_epochs': 3,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'inference_batch_size': 16,  # 4
    'evaluate_every': 20,  # 4
    'learning_rate': 1e-5,  # 3e-5
    'metric': ["f1_macro", "acc"],
    'use_early_stopping': True,
    'early_stopping_metric': "loss",
    'early_stopping_mode': "min",  # "min" or "max"
    'early_stopping_patience': 10,

    'label_list': ["0", "1"],
    # 'train_filename': "en-train.tsv",  # given languages, this will be automatically set
    'dev_split': 0.1,
    'dev_stratification': True,
    'delimiter': "\t",
    'label_column_name': "label",  # for classification
    'text_column_name': "text",
    'max_processes': 128,  # 128 is default
    'use_amp': None,
    'cude_device': "2",  # "1",  # or None

    'n_fold': 5,
    'fold_ids': [0, 1, 2, 3, 4],  # list of ids for folds

    # for inferencer
    'gpu': True,
    'num_processes': cpu_count() - 2 if cpu_count() > 2 else 1
}


