# Created by Hansi at 9/28/2021
import os

import numpy as np

from algo.models.classification_model import ClassificationModel
from algo.util.file_util import delete_create_folder
from experiments.classifier_config import MODEL_NAME, config, DATA_DIRECTORY, OUTPUT_DIRECTORY
from farm.utils import set_all_seeds

if __name__ == '__main__':
    delete_create_folder(OUTPUT_DIRECTORY)

    test_sentences = [  # 0, 1
        {"text": "The movement is led by Benny Tai Yiu-ting, an assistant law professor at the University of Hong Kong"},
        {"text": "A protest last Wednesday, organised by Hong Kong international students against the controversial extradition law, turned violent."},
    ]
    test_preds = np.zeros((len(test_sentences), config["n_fold"]))

    for i in range(config["n_fold"]):
        config['model_dir'] = f"{config['model_dir']}_{i}"
        train_progress_file_name = os.path.basename(config['train_progress_file'])
        file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(os.path.dirname(config['train_progress_file']),
                                                     f"{file_name_splits[0]}_{i}{file_name_splits[1]}")

        set_all_seeds(seed=config['manual_seed'] * (i + 1))

        model = ClassificationModel(MODEL_NAME, args=config)
        data_dir = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/filtered/temp')
        model.train_model(data_dir)

        # basic_texts = [ # 0, 1
        #     {"text": "The movement is led by Benny Tai Yiu-ting, an assistant law professor at the University of Hong Kong"},
        #     {"text": "A protest last Wednesday, organised by Hong Kong international students against the controversial extradition law, turned violent."},
        # ]

        predictions, raw_predictions = model.predict(test_sentences)
        print(predictions)
        print(f'raw predictions: {raw_predictions}')

        test_preds[:, i] = predictions

    print(test_preds)

    # select majority class of each instance (row)
    test_predictions = []
    for row in test_preds:
        row = row.tolist()
        test_predictions.append(int(max(set(row), key=row.count)))
    # test["predictions"] = test_predictions
    print(test_predictions)
