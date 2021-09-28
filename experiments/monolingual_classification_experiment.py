# Created by Hansi at 9/28/2021
import os

from algo.models.classification_model import ClassificationModel
from algo.util.file_util import delete_create_folder
from experiments.classifier_config import MODEL_NAME, config, DATA_DIRECTORY, OUTPUT_DIRECTORY

if __name__ == '__main__':
    delete_create_folder(OUTPUT_DIRECTORY)
    model = ClassificationModel(MODEL_NAME, args=config)
    data_dir = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/filtered/temp')
    model.train_model(data_dir)

    basic_texts = [ # 0, 1
        {"text": "The movement is led by Benny Tai Yiu-ting, an assistant law professor at the University of Hong Kong"},
        {"text": "A protest last Wednesday, organised by Hong Kong international students against the controversial extradition law, turned violent."},
    ]

    preds = model.predict(basic_texts)
    print(preds)


