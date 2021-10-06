# Created by Hansi at 10/4/2021
import logging
import os

import numpy as np

from algo.models.ner_model import NERModel
from algo.util.file_util import delete_create_folder
from experiments.data_process.data_util import read_tokens, get_token_test_instances
from experiments.ner_config import OUTPUT_DIRECTORY, DATA_DIRECTORY, config, MODEL_NAME, SUBMISSION_FILE, LANGUAGES
from farm.utils import set_all_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def majority_class_for_ner(sentences, preds, n_folds):
    """
    Get majority class label for NER

    :param sentences: list of dict {'text': "sample text"}
    :param preds: list
        predictions (IOB2 format) by all folds - [[fold_0 predictions], ... [fold_n predictions]]
    :return: list
        majority class IOB2 formatted predictions
    """
    preds = np.array(preds)
    final_preds = []
    # print(test_preds)
    for n in range(len(sentences)):  # iterate through each sentence
        # print(f'sentence: {sentences[n]}')
        temp_preds = []
        # print(f'{preds[:, n]}')
        for i in range(len(preds[:, n][0])):  # iterate through each token
            fold_preds = [preds[:, n][k][i] for k in range(n_folds)]  # get predictions by folds for token
            # print(f'{i} - {fold_preds}')
            temp_preds.append(
                max(set(fold_preds), key=fold_preds.count))  # get majority class and add to temp_predictions
        final_preds.append(temp_preds)
    return final_preds


if __name__ == '__main__':
    delete_create_folder(OUTPUT_DIRECTORY)
    # data_dir = os.path.join(DATA_DIRECTORY, 'conll2003/temp')
    data_dir = os.path.join(DATA_DIRECTORY, 'subtask4-token/filtered/farm_format')

    # read test data
    test_data_path = os.path.join(DATA_DIRECTORY, "subtask4-token", f"{LANGUAGES[0]}-test.txt")
    test_tokens, _ = read_tokens(test_data_path, train=False)
    sentence_tokens, dict_instance_sentence = get_token_test_instances(test_tokens)
    logger.info(f"{len(sentence_tokens)} test sentences are loaded.")
    test_sentences = [{"text": " ".join(sent_tokens)} for sent_tokens in sentence_tokens]

    # test_sentences = [
    #     {"text": "Paris is a town in France ."},
    #     {"text": "Only France and Britain backed proposal ."},
    #     {"text": "Peter Blackburn"},
    # ]
    # test_tokens = [sent["text"].split() for sent in test_sentences]

    test_preds = []  # [[fold_0 predictions], ... [fold_n predictions]]

    for i in range(config["n_fold"]):
        # update model dir for the fold
        config['model_dir'] = f"{config['model_dir']}_{i}"
        train_progress_file_name = os.path.basename(config['train_progress_file'])
        file_name_splits = os.path.splitext(train_progress_file_name)
        config['train_progress_file'] = os.path.join(os.path.dirname(config['train_progress_file']),
                                                     f"{file_name_splits[0]}_{i}{file_name_splits[1]}")

        set_all_seeds(seed=config['manual_seed'] * (i + 1))

        model = NERModel(MODEL_NAME, args=config)
        logger.info(f"Training model for fold {i}...")
        model.train_model(data_dir)

        logger.info(f"Making test predictions for fold {i}...")
        predictions, raw_predictions = model.predict(test_sentences, config['inference_batch_size'])
        # print(raw_predictions)
        test_preds.append(predictions)

    # select majority class for each token in each sentence
    logger.info(f"Calculating majority class...")
    final_preds = majority_class_for_ner(test_sentences, test_preds, config["n_fold"])

    # merge split sentences
    merged_preds = []
    for k, v in dict_instance_sentence.items():
        merged_pred = final_preds[v[0]]  # There is at least 1 sentence in an instance
        if len(v) > 1:
            for i in v[1:len(v)]:
                merged_pred.extend(["O"])
                merged_pred.extend(final_preds[i])
        merged_preds.append(merged_pred)

    logger.info(f"Saving test predictions...")
    with open(SUBMISSION_FILE, "w", encoding="utf-8") as f:
        for tokens, labels in zip(test_tokens, merged_preds):
            f.write("SAMPLE_START\tO\n")
            for token, label in zip(tokens, labels):
                f.write("{}\t{}\n".format(token, label))
            f.write("\n")

