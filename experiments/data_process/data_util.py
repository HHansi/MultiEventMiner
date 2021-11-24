# Created by Hansi at 4/9/2021

import json
import os
import re

import pandas as pd

from algo.util.data_preprocessor import remove_links
from algo.util.file_util import create_folder_if_not_exist
from experiments.ner_config import DATA_DIRECTORY
from farm.data_handler.utils import read_ner_file


def read_data(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data


def read_data_df(path):
    data = read_data(path)
    return pd.DataFrame.from_records(data)


def save_data(data, path):
    create_folder_if_not_exist(path, is_file_path=True)
    with open(path, 'w') as f:
        for i in data:
            f.write("%s\n" % json.dumps(i))


def read_tokens(path, train=True):
    """
    Reads the file from the given path (txt file).
    Returns list tokens and list of labels if it is training file.
    Returns list of tokens if it is test file.
    """
    with open(path, 'r', encoding="utf-8") as f:
        data = f.read()

    if train:
        data = [[tuple(word.split('\t')) for word in instance.strip().split('\n')] for idx, instance in
                enumerate(data.split("SAMPLE_START\tO")) if len(instance) > 1]
        tokens = [[tupl[0].strip() for tupl in sent] for sent in data]
        labels = [[tupl[1] for tupl in sent] for sent in data]
        return tokens, labels
    else:
        tokens = [[word for word in instance.strip().split('\n')] for idx, instance in
                  enumerate(data.split("SAMPLE_START")) if len(instance) > 1]
        return tokens, None


def read_tokens_farm_format(path):
    data = read_ner_file(path)
    tokens = [x['text'].split() for x in data]
    labels = [x['ner_label'] for x in data]
    return tokens, labels


def save_tokens(tokens, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in zip(tokens, labels):
            f.write("SAMPLE_START\tO\n")
            for token, label in zip(tokens, labels):
                f.write("{}\t{}\n".format(token, label))
            f.write("\n")


def save_tokens_farm_format(tokens, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("-DOCSTART-\tO\n\n")
        for temp_tokens, temp_labels in zip(tokens, labels):
            for token, label in zip(temp_tokens, temp_labels):
                f.write("{}\t{}\n".format(token, label))
            f.write("\n")


def preprocess_data(text):
    text = text.replace("\n", " ")
    text = re.sub("__+", "", text)  # remove if text has more than 1 _
    text = remove_links(text, substitute='')
    # remove white spaces at the beginning and end of the text
    text = text.strip()
    # remove extra whitespace, newline, tab
    text = ' '.join(text.split())
    return text


def get_token_test_instances(tokens):
    """
    Load test sentences (NER)
    :param tokens: list of tokens
    :return: list, dict,
        list of sentence tokens
        dict {instance_index: [sentence indices in list of sentence tokens]}
    """
    sentence_tokens = []
    dict_instance_sentence = dict()
    sentence_index = -1

    for idx, temp_tokens in enumerate(tokens):
        instance_sentence_indices = []
        SEP_indices = [i for i, value in enumerate(temp_tokens) if value == '[SEP]']

        if len(SEP_indices) == 0:  # If no [SEP] labels found, the instance has one sentence.
            sentence_tokens.append(temp_tokens)
            sentence_index += 1
            instance_sentence_indices.append(sentence_index)
        else:
            SEP_indices.insert(0, -1)  # Add the index of -1 to the beginning
            SEP_indices.insert(len(SEP_indices), len(temp_tokens))

            for i in range(0, len(SEP_indices)):
                if i < (len(SEP_indices) - 1):
                    temp_sent_tokens = temp_tokens[SEP_indices[i] + 1:SEP_indices[i + 1]]

                    sentence_tokens.append(temp_sent_tokens)
                    sentence_index += 1
                    instance_sentence_indices.append(sentence_index)
        dict_instance_sentence[idx] = instance_sentence_indices

    return sentence_tokens, dict_instance_sentence


if __name__ == '__main__':
    # test_data_path = os.path.join(DATA_DIRECTORY, "subtask4-token", "en-test.txt")
    # tokens, _ = read_tokens(test_data_path, train=False)
    # sentence_tokens, dict_instance_sentence= get_token_test_instances(tokens)
    # print(len(sentence_tokens))
    #
    # merged = []
    # for k, v in dict_instance_sentence.items():
    #     merged_pred = sentence_tokens[v[0]]
    #     if len(v) > 1:
    #         for i in v[1:len(v)]:
    #             merged_pred.extend(["O"])
    #             merged_pred.extend(sentence_tokens[i])
    #     merged.append(merged_pred)
    # print(merged)

    path = os.path.join(DATA_DIRECTORY, "subtask4-token", "filtered/farm_format/en-en-en-train.txt")
    data = read_ner_file(path)
    print()



