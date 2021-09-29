# Created by Hansi at 4/9/2021

import json
import re

import pandas as pd

from algo.data_process.data_preprocessor import remove_links
from algo.util.file_util import create_folder_if_not_exist


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


def save_tokens(tokens, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in zip(tokens, labels):
            f.write("SAMPLE_START\tO\n")
            for token, label in zip(tokens, labels):
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



