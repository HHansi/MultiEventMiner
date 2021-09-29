# Created by Hansi at 9/28/2021
import logging
import os

import numpy as np

from algo.models.classification_model import ClassificationModel
from algo.util.file_util import delete_create_folder
from experiments.classifier_config import MODEL_NAME, config, DATA_DIRECTORY, OUTPUT_DIRECTORY, LANGUAGES, \
    SUBMISSION_FILE
from experiments.data_process.data_util import read_data_df, preprocess_data
from farm.utils import set_all_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    delete_create_folder(OUTPUT_DIRECTORY)
    data_dir = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/filtered/farm_format')

    # read test data
    test_data_path = os.path.join(DATA_DIRECTORY, "subtask2-sentence", f"{LANGUAGES[0]}-test.json")
    test_df = read_data_df(test_data_path)
    test_df = test_df.rename(columns={'sentence': 'text'})
    test_df['text'] = test_df['text'].apply(lambda x: preprocess_data(x))
    test_df = test_df.head(100)
    test_sentences = test_df[['text']].to_dict(orient='records')

    # test_sentences = [  # 0, 1
    #     {"text": "The movement is led by Benny Tai Yiu-ting, an assistant law professor at the University of Hong Kong"},
    #     {"text": "A protest last Wednesday, organised by Hong Kong international students against the controversial extradition law, turned violent."},
    # ]
    test_preds = np.zeros((len(test_sentences), config["n_fold"]))

    for i in range(config["n_fold"]):
        # update model dir for the fold
        config['model_dir'] = f"{config['model_dir']}_{i}"
        train_progress_file_name = os.path.basename(config['train_progress_file'])
        file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(os.path.dirname(config['train_progress_file']),
                                                     f"{file_name_splits[0]}_{i}{file_name_splits[1]}")

        set_all_seeds(seed=config['manual_seed'] * (i + 1))

        model = ClassificationModel(MODEL_NAME, args=config)
        logger.info(f"Training model for fold {i}...")
        model.train_model(data_dir)

        logger.info(f"Making test predictions for fold {i}...")
        predictions, raw_predictions = model.predict(test_sentences, config['inference_batch_size'])
        test_preds[:, i] = predictions

    # select majority class of each instance (row)
    logger.info(f"Calculating majority class...")
    test_predictions = []
    for row in test_preds:
        row = row.tolist()
        test_predictions.append(int(max(set(row), key=row.count)))
    test_df["predictions"] = test_predictions

    logger.info(f"Saving test predictions...")
    with open(SUBMISSION_FILE, 'w') as f:
        for index, row in test_df.iterrows():
            item = {"id": row['id'], "prediction": row['predictions']}
            f.write("%s\n" % item)
