# Created by Hansi at 8/12/2022
from algo.models.classification_model import ClassificationModel
from experiments import classifier_config
from experiments.classifier_config import config

import time
import os, psutil


def sentence_predict():
    text_en = "A child ran around in a T-shirt that read: New Great Country Wonderful Country China."
    text_es = "Por eso aquel artículo de Bayer me pareció por lo menos temerario."
    text_pr = "Investigadores das duas unidades devem viajar em breve ao Paquistão, afirmou a fonte."

    texts = [{'text': text_pr}]

    # set cuda device
    if config["cude_device"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cude_device"]

    config['model_dir'] = "/experiments/tranasinghe/MultiEventMiner/trained_models2/sentence/bertimbau-large-pr/model_0"

    print(f'loading model')
    start_time = time.time()
    model = ClassificationModel(classifier_config.MODEL_NAME, args=config, mode='inference')
    end_time = time.time()
    print(f'Model loaded in {(end_time - start_time)} seconds \n')
    print(f'loaded: {classifier_config.MODEL_NAME}')

    # print(f'sleeping')
    # time.sleep(10)

    process = psutil.Process(os.getpid())
    print(f'RSS: {process.memory_info().rss / 1024 ** 2}')
    print(f'VMS: {process.memory_info().vms / 1024 ** 2}')

    print(f'predicting')
    start_time = time.time()
    predictions, raw_predictions = model.predict(texts)
    end_time = time.time()
    print(f'Predicted in {(end_time - start_time)} seconds \n')

    print(f'predictions: {predictions}')
    print(f'raw predictions: {raw_predictions}')


if __name__ == '__main__':
    sentence_predict()