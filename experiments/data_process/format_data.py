# Created by Hansi at 9/21/2021
import ast
import copy
import logging
import os

import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import shuffle
from tqdm import tqdm

from algo.util.file_util import create_folder_if_not_exist
from experiments.data_process.analyse_data import plot_seq_length_histo
from experiments.data_process.clean_data import get_token_sentences, get_token_level_duplicates
from experiments.data_process.data_util import read_data_df, preprocess_data, read_tokens, save_tokens_farm_format
from experiments.data_config import DATA_DIRECTORY, SEED, SENTENCE_TRAIN_DATA_FILE, SENTENCE_DEV_DATA_FILE, \
    TOKEN_TRAIN_DATA_FILE, TOKEN_DEV_DATA_FILE
from farm.data_handler.utils import read_ner_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sentence_to_tsv(input_path, output_path):
    df = read_data_df(input_path)
    # df = df[['sentence', 'label']]
    df = df.rename(columns={'sentence': 'text'})
    df['text'] = df['text'].apply(lambda x: preprocess_data(x))
    df['text'].replace('', np.nan, inplace=True)
    df.dropna()
    # df = df.head(250)
    print(f'Number of instances: {df.shape[0]}')
    df.to_csv(output_path, sep='\t', index=False)


def token_to_ner(input_path, output_path, train=True):
    tokens, labels = read_tokens(input_path, train=train)
    token_data, sentence_tokens, sentence_labels, token_sentence_count, sentence_without_trigger_count = \
        get_token_sentences(tokens, labels)
    save_tokens_farm_format(sentence_tokens, sentence_labels, output_path)


def doc_to_lm_data(doc_train_path, doc_test_path, output_path, plot_path=None):
    doc_train = read_data_df(doc_train_path)
    doc_test = read_data_df(doc_test_path)
    docs = doc_train["text"].tolist()
    docs.extend(doc_test["text"].tolist())
    print(f"{len(docs)} documents are loaded.")
    all_sentences = []
    i = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in tqdm(docs):
            i += 1
            sentences = sent_tokenize(doc)
            all_sentences.extend(sentences)
            for sent in sentences:
                if not preprocess_data(sent):
                    print('yes')
                f.write(f"{preprocess_data(sent)}\n")
            f.write("\n")
            # if i == 250:
            #     break
    print(i)
    print(f"{len(all_sentences)} sentences are saved.")
    plot_seq_length_histo(all_sentences, plot_path=plot_path)


def sentence_to_lm_data(sent_path, output_path, plot_path=None):
    sent_data = read_data_df(sent_path)
    sentences = sent_data["sentence"].tolist()
    print(f"{len(sentences)} sentences are loaded.")
    i = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sent in tqdm(sentences):
            preprocessed = preprocess_data(sent)
            if len(preprocessed) > 0:
                i += 1
                f.write(f"{preprocessed}\n")
                f.write("\n")

    print(f"{i} sentences are saved.")
    plot_seq_length_histo(sentences, plot_path=plot_path)


def split_tokens(input_path, output_folder, test_size=0.2, binary_labels=False, sentence_data_path=None):
    """
    Split token data into train and test splits

    :param input_path: .txt file
        data in farm format
    :param output_folder:
    :param test_size: float, optional
        test split size
    :param binary_labels: boolean, optional
        True - convert IOB labels to binary (0 and 1)
    :param sentence_data_path: file path, optional
        sentence data file path
        If given, token split will be created without any duplicates from sentence data
    :return:
    """
    create_folder_if_not_exist(output_folder)
    data = read_ner_file(input_path)
    # train, dev = train_test_split(data, shuffle=True, test_size=test_size, random_state=157)

    if sentence_data_path:
        targeted_dev_size = int(round(len(data)*0.2))
        print(f"Targeted dev size = {targeted_dev_size}")
        sent_df = read_data_df(sentence_data_path)
        tokens = [x['text'].split() for x in data]
        duplicate_ids = get_token_level_duplicates(tokens, sent_df)
        print(f"{len(duplicate_ids)} duplicates are found.")
        print(duplicate_ids)

        duplicates = []
        for id in duplicate_ids:
            duplicates.append(data[id])
            # del non_duplicates[id]

        non_duplicates = copy.deepcopy(data)
        for element in duplicates:
            non_duplicates.remove(element)
        print(f"{len(non_duplicates)} non-duplicates are found.")

        if len(non_duplicates) < targeted_dev_size:
            raise ValueError(f"No enough non duplicate data!")
        train, dev = train_test_split(non_duplicates, shuffle=True, test_size=targeted_dev_size, random_state=SEED)
        train = train + duplicates

    else:
        train, dev = train_test_split(data, shuffle=True, test_size=test_size, random_state=SEED)

    print(f"Train size: {len(train)}")
    print(f"Dev size: {len(dev)}")

    train_sentences = [x['text'].split() for x in train]
    train_labels = [x['ner_label'] for x in train]
    dev_sentences = [x['text'].split() for x in dev]
    dev_labels = [x['ner_label'] for x in dev]

    for idx, sent in enumerate(train_sentences):
        if len(sent) != len(train_labels[idx]):
            raise IndexError(f"Train sentence and label mismatch!")
    for idx, sent in enumerate(dev_sentences):
        if len(sent) != len(dev_labels[idx]):
            raise IndexError(f"Dev sentence and label mismatch!")

    train_split_path = os.path.join(output_folder, "en-train.txt")
    dev_split_path = os.path.join(output_folder, "en-dev.txt")

    if binary_labels:
        binary_train_labels = []
        for ls in train_labels:
            labels = [0 if x == 'O' else 1 for x in ls]
            binary_train_labels.append(labels)
        binary_dev_labels = []
        for ls in dev_labels:
            labels = [0 if x == 'O' else 1 for x in ls]
            binary_dev_labels.append(labels)
        save_tokens_farm_format(train_sentences, binary_train_labels, train_split_path)
        save_tokens_farm_format(dev_sentences, binary_dev_labels, dev_split_path)
    else:
        save_tokens_farm_format(train_sentences, train_labels, train_split_path)
        print(f"Saved {len(train_sentences)} train instances.")
        save_tokens_farm_format(dev_sentences, dev_labels, dev_split_path)
        print(f"Saved {len(dev_sentences)} dev instances.")


def format_multi_task_data(sentence_data_path, token_data_path, output_path):
    df = pd.DataFrame(columns=['text', 'text_tokens', 'token_label_iob', 'token_label_binary', 'text_label'])
    data = read_ner_file(token_data_path)
    df_index = 0
    for i, element in tqdm(enumerate(data)):
        binary_labels = [0 if x == 'O' else 1 for x in element['ner_label']]
        df.loc[i] = [element['text'], element['text'].split(), element['ner_label'], binary_labels, 1]
        df_index = i

    sent_df = read_data_df(sentence_data_path)
    sent_df_filtered = sent_df.loc[sent_df['label'] == 0]
    sent_df_filtered['seq_length'] = [len(word_tokenize(x)) for x in sent_df_filtered['sentence']]
    sent_df_filtered = sent_df_filtered.loc[sent_df_filtered['seq_length'] > 4]
    n = len(data)
    sent_df_sample = sent_df_filtered.sample(n=n)
    j = 0
    for index, row in tqdm(sent_df_sample.iterrows()):
        j += 1
        i = df_index + j
        tokens = word_tokenize(row['sentence'])
        token_label_iob = ['O' for x in tokens]
        token_label_binary = [0 for x in tokens]
        df.loc[i] = [' '.join(tokens), tokens, token_label_iob, token_label_binary, 0]

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


def split_sentence_data(data_file_path, seed, args, output_folder=None):
    data = pd.read_csv(data_file_path, sep=args['delimiter'])
    if args['dev_stratification']:
        if args['label_column_name'] is None:
            raise ValueError(f"Label column should be provided for dev stratification!")
        y = data[args['label_column_name']]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args['dev_split'], random_state=seed)
        train_index, test_index = next(sss.split(data, y))

        train = data.iloc[train_index]
        dev = data.iloc[test_index]
    else:
        train, dev = train_test_split(data, test_size=args['dev_split'], random_state=seed)

    logger.info(f"Train instances: {train.shape[0]}")
    logger.info(f"Dev instances: {dev.shape[0]}")
    if output_folder:
        train.to_csv(os.path.join(output_folder, SENTENCE_TRAIN_DATA_FILE), sep=args['delimiter'], index=False)
        logger.info(f"Saved {train.shape[0]} train instances.")
        dev.to_csv(os.path.join(output_folder, SENTENCE_DEV_DATA_FILE), sep=args['delimiter'], index=False)
        logger.info(f"Saved {dev.shape[0]} dev instances.")
    return train, dev


def split_token_data(data_file_path, seed, args, output_folder=None):
    data = read_ner_file(data_file_path)
    train, dev = train_test_split(data, test_size=args['dev_split'], random_state=seed)

    logger.info(f"Train instances: {len(train)}")
    logger.info(f"Dev instances: {len(dev)}")
    if output_folder:
        train_sentences = [x['text'].split() for x in train]
        train_labels = [x['ner_label'] for x in train]
        dev_sentences = [x['text'].split() for x in dev]
        dev_labels = [x['ner_label'] for x in dev]

        for idx, sent in enumerate(train_sentences):
            if len(sent) != len(train_labels[idx]):
                raise IndexError(f"Train sentence and label mismatch!")
        for idx, sent in enumerate(dev_sentences):
            if len(sent) != len(dev_labels[idx]):
                raise IndexError(f"Dev sentence and label mismatch!")

        save_tokens_farm_format(train_sentences, train_labels, os.path.join(output_folder, TOKEN_TRAIN_DATA_FILE))
        logger.info(f"Saved {len(train_sentences)} train instances.")
        save_tokens_farm_format(dev_sentences, dev_labels, os.path.join(output_folder, TOKEN_DEV_DATA_FILE))
        logger.info(f"Saved {len(dev_sentences)} dev instances.")
    return train, dev


def prepare_multilingual_sentence_data(languages, input_folder, seed, args, output_folder=None):
    train = pd.DataFrame(columns=['id', 'label', 'text'])
    dev = pd.DataFrame(columns=['id', 'label', 'text'])

    for lang in languages:
        logger.info(f"Splitting {lang}..")
        file_path = os.path.join(input_folder, f"{lang}-train.tsv")
        temp_train, temp_dev = split_sentence_data(file_path, seed, args, output_folder=None)
        train = train.append(temp_train)
        dev = dev.append(temp_dev)

    # shuffle train and dev sets
    train = shuffle(train, random_state=seed)
    dev = shuffle(dev, random_state=seed)

    if output_folder:
        train.to_csv(os.path.join(output_folder, SENTENCE_TRAIN_DATA_FILE), sep=args['delimiter'], index=False)
        logger.info(f"Saved {train.shape[0]} train instances.")
        dev.to_csv(os.path.join(output_folder, SENTENCE_DEV_DATA_FILE), sep=args['delimiter'], index=False)
        logger.info(f"Saved {dev.shape[0]} dev instances.")
    return train, dev


def prepare_multilingual_token_data(languages, input_folder, seed, args, output_folder=None):
    train = []
    dev = []

    for lang in languages:
        logger.info(f"Splitting {lang}..")
        file_path = os.path.join(input_folder, f"{lang}-train.txt")
        temp_train, temp_dev = split_token_data(file_path, seed, args, output_folder=None)
        train.extend(temp_train)
        dev.extend(temp_dev)

    # shuffle train and dev sets
    train = shuffle(train, random_state=seed)
    dev = shuffle(dev, random_state=seed)

    if output_folder:
        train_sentences = [x['text'].split() for x in train]
        train_labels = [x['ner_label'] for x in train]
        dev_sentences = [x['text'].split() for x in dev]
        dev_labels = [x['ner_label'] for x in dev]

        for idx, sent in enumerate(train_sentences):
            if len(sent) != len(train_labels[idx]):
                raise IndexError(f"Train sentence and label mismatch!")
        for idx, sent in enumerate(dev_sentences):
            if len(sent) != len(dev_labels[idx]):
                raise IndexError(f"Dev sentence and label mismatch!")

        save_tokens_farm_format(train_sentences, train_labels, os.path.join(output_folder, TOKEN_TRAIN_DATA_FILE))
        logger.info(f"Saved {len(train_sentences)} train instances.")
        save_tokens_farm_format(dev_sentences, dev_labels, os.path.join(output_folder, TOKEN_DEV_DATA_FILE))
        logger.info(f"Saved {len(dev_sentences)} dev instances.")
    return train, dev


if __name__ == '__main__':
    input_path = os.path.join(DATA_DIRECTORY, '../data/subtask2-sentence/filtered/en-train.json')
    output_path = os.path.join(DATA_DIRECTORY, '../data/subtask2-sentence/filtered/farm_format/en-train2.tsv')
    sentence_to_tsv(input_path, output_path)

    # input_path = os.path.join(DATA_DIRECTORY, 'subtask4-token/filtered/es-train.txt')
    # output_path = os.path.join(DATA_DIRECTORY, 'subtask4-token/filtered/farm_format/es-train.txt')
    # token_to_ner(input_path, output_path, train=True)

    # doc_train_path = os.path.join(DATA_DIRECTORY, 'subtask1-doc/en-train.json')
    # doc_test_path = os.path.join(DATA_DIRECTORY, 'subtask1-doc/en-test.json')
    # output_path = os.path.join(DATA_DIRECTORY, 'subtask1-doc/en-lm2.txt')
    # plot_path = os.path.join(DATA_DIRECTORY, 'subtask1-doc/plots/en-lm.png')
    # doc_to_lm_data(doc_train_path, doc_test_path, output_path, plot_path=None)

    # sent_path = os.path.join(DATA_DIRECTORY, '../data/subtask2-sentence/filtered/en-train.json')
    # output_path = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/en-lm.txt')
    # sentence_to_lm_data(sent_path, output_path, plot_path=None)

    # input_path = os.path.join(DATA_DIRECTORY, "subtask4-token", "filtered/farm_format/en-train.txt")
    # output_folder = os.path.join(DATA_DIRECTORY, "subtask4-token", "filtered/farm_format/split2")
    # sentence_data_path = os.path.join(DATA_DIRECTORY, "subtask2-sentence", "filtered/en-train.json")
    # split_tokens(input_path, output_folder, test_size=0.2, binary_labels=False, sentence_data_path=sentence_data_path)

    # sentence_data_path = os.path.join(DATA_DIRECTORY, "subtask2-sentence", "filtered/en-train.json")
    # token_data_path = os.path.join(DATA_DIRECTORY, "subtask4-token", "filtered/farm_format/split_binary/en-train.txt")
    # output_path = os.path.join(DATA_DIRECTORY, "joint_data", "en-train-binary.csv")
    # format_multi_task_data(sentence_data_path, token_data_path, output_path)




