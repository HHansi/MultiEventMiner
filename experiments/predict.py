# Created by Hansi at 10/11/2021
import logging
import os
import re

import numpy as np

from algo.models.classification_model import ClassificationModel
from algo.models.ner_model import NERModel
from algo.util.file_util import create_folder_if_not_exist
from experiments import classifier_config
from experiments import ner_config
from experiments.data_process.data_util import read_data_df, preprocess_data, read_tokens, get_token_test_instances
from experiments.ner_experiment import majority_class_for_ner
from farm.utils import set_all_seeds
# import warnings
# warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestInstanceClassifier:
    def __init__(self, lang, df, sentences, preds):
        self.lang = lang
        self.df = df
        self.sentences = sentences
        self.preds = preds


class TestInstanceNER:
    def __init__(self, lang, test_tokens, dict_instance_sentence, sentences, preds):
        self.lang = lang
        self.test_tokens = test_tokens
        self.dict_instance_sentence = dict_instance_sentence
        self.sentences = sentences
        self.preds = preds


def remove_ib(iob_outputs):
    """
    Remove I- and B- tags in IOB labels
    :param iob_outputs: list
        list of list which contains IOB tags
    :return: list
        list of list only with labels except I- and B-
    """
    updated_outputs = []
    for sent in iob_outputs:
        updated = [re.sub("^[BI]-", "", token) for token in sent]
        updated_outputs.append(updated)
    return updated_outputs


def labels_to_iob(labels):
    """
    Covert labels to IOB2 format
    :param labels: list
        list of list of labels (O and other labels without I- and B-)
    :return: list
        list of list which contains IOB tags
    """
    iob_labels = []
    for instance in labels:
        previous_label = ""
        iob = []
        for label in instance:
            if label == "O":
                updated_label = label
            else:
                if label == previous_label:
                    updated_label = f"I-{label}"
                else:
                    updated_label = f"B-{label}"
            iob.append(updated_label)
            previous_label = label
        iob_labels.append(iob)
    return iob_labels


def predict_classifier(args):
    create_folder_if_not_exist(classifier_config.PREDICTION_DIRECTORY)
    test_instances = dict()
    for lang in classifier_config.LANGUAGES:
        # read test data
        test_data_path = os.path.join(classifier_config.DATA_DIRECTORY, "subtask2-sentence", f"{lang}-test.json")
        test_df = read_data_df(test_data_path)
        test_df = test_df.rename(columns={'sentence': 'text'})
        test_df['text'] = test_df['text'].apply(lambda x: preprocess_data(x))
        test_sentences = test_df[['text']].to_dict(orient='records')
        logger.info(f"{len(test_sentences)} test sentences are loaded.")

        test_preds = np.zeros((len(test_sentences), args["n_fold"]))
        test_instances[lang] = TestInstanceClassifier(lang, test_df, test_sentences, test_preds)

    base_model_dir = args['model_dir']
    for i in range(args["n_fold"]):
        args['model_dir'] = f"{base_model_dir}_{i}"

        set_all_seeds(seed=int(args['manual_seed'] * (i + 1)))
        logger.info(f"Set seed to {int(args['manual_seed'] * (i + 1))}")

        model = ClassificationModel(classifier_config.MODEL_NAME, args=args)

        logger.info(f"Making test predictions for fold {i}...")
        for lang in test_instances.keys():
            predictions, raw_predictions = model.predict(test_instances[lang].sentences)
            test_instances[lang].preds[:, i] = predictions

        del model

    for lang in test_instances.keys():
        logger.info(f"Calculating majority class for {lang}...")
        test_predictions = []
        for row in test_instances[lang].preds:
            row = row.tolist()
            test_predictions.append(int(max(set(row), key=row.count)))  # select majority class of each instance (row)
        test_instances[lang].df["predictions"] = test_predictions

        logger.info(f"Saving test predictions for {lang}...")
        submission_file_name = os.path.basename(classifier_config.SUBMISSION_FILE)
        submission_file_name_splits = os.path.splitext(submission_file_name)
        submission_file = os.path.join(os.path.dirname(classifier_config.SUBMISSION_FILE),
                                       f"{submission_file_name_splits[0]}_{lang}{submission_file_name_splits[1]}")
        with open(submission_file, 'w') as f:
            for index, row in test_instances[lang].df.iterrows():
                item = {"id": row['id'], "prediction": row['predictions']}
                f.write("%s\n" % item)


def predict_ner(args):
    create_folder_if_not_exist(ner_config.PREDICTION_DIRECTORY)
    test_instances = dict()
    for lang in ner_config.LANGUAGES:
        # read test data
        test_data_path = os.path.join(ner_config.DATA_DIRECTORY, "subtask4-token", f"{lang}-test.txt")
        test_tokens, _ = read_tokens(test_data_path, train=False)
        sentence_tokens, dict_instance_sentence = get_token_test_instances(test_tokens)
        logger.info(f"{len(sentence_tokens)} test sentences are loaded.")
        test_sentences = [{"text": " ".join(sent_tokens)} for sent_tokens in sentence_tokens]

        test_preds = []  # [[fold_0 predictions], ... [fold_n predictions]]
        test_instances[lang] = TestInstanceNER(lang, test_tokens, dict_instance_sentence, test_sentences, test_preds)

    base_model_dir = args['model_dir']
    for i in range(args["n_fold"]):
        args['model_dir'] = f"{base_model_dir}_{i}"

        set_all_seeds(seed=int(args['manual_seed'] * (i + 1)))
        logger.info(f"Set seed to {int(args['manual_seed'] * (i + 1))}")

        model = NERModel(ner_config.MODEL_NAME, args=args)

        logger.info(f"Making test predictions for fold {i}...")
        for lang in test_instances.keys():
            predictions, raw_predictions = model.predict(test_instances[lang].sentences)
            test_instances[lang].preds.append(remove_ib(predictions))

        del model

    for lang in test_instances.keys():
        # select majority class for each token in each sentence
        logger.info(f"Calculating majority class for {lang}...")
        final_preds = majority_class_for_ner(test_instances[lang].sentences, test_instances[lang].preds, args["n_fold"])
        final_preds = labels_to_iob(final_preds)

        # merge split sentences
        merged_preds = []
        for k, v in test_instances[lang].dict_instance_sentence.items():
            merged_pred = final_preds[v[0]]  # There is at least 1 sentence in an instance
            if len(v) > 1:
                for i in v[1:len(v)]:
                    merged_pred.extend(["O"])
                    merged_pred.extend(final_preds[i])
            merged_preds.append(merged_pred)

        logger.info(f"Saving test predictions for {lang}...")
        submission_file_name = os.path.basename(ner_config.SUBMISSION_FILE)
        submission_file_name_splits = os.path.splitext(submission_file_name)
        submission_file = os.path.join(os.path.dirname(ner_config.SUBMISSION_FILE),
                                       f"{submission_file_name_splits[0]}_{lang}{submission_file_name_splits[1]}")
        with open(submission_file, "w", encoding="utf-8") as f:
            for tokens, labels in zip(test_instances[lang].test_tokens, merged_preds):
                f.write("SAMPLE_START\tO\n")
                for token, label in zip(tokens, labels):
                    f.write("{}\t{}\n".format(token, label))
                f.write("\n")


if __name__ == '__main__':
    predict_classifier(classifier_config.config)

    # predict_ner(ner_config.config)
