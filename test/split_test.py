# Created by Hansi at 12/9/2021
import os
import unittest

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from experiments import classifier_config, ner_config
from experiments.data_process.format_data import split_sentence_data, prepare_multilingual_sentence_data, \
    split_token_data


class TestSplits(unittest.TestCase):
    def test_shuffle_split_df(self):
        data_file = os.path.join(classifier_config.DATA_DIRECTORY,
                                 'subtask2-sentence/filtered/farm_format/en-train.tsv')
        print(data_file)
        data = pd.read_csv(data_file, sep='\t')
        y = data["label"]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_index, test_index = next(sss.split(data, y))
        print("TRAIN:", train_index, "TEST:", test_index)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))

        train_df1 = data.iloc[train_index]
        test_df1 = data.iloc[test_index]
        print(train_df1.shape)
        print(test_df1.shape)

        temp_train_df1 = train_df1.groupby('label').count()
        print(temp_train_df1)
        temp_test_df1 = test_df1.groupby('label').count()
        print(temp_test_df1)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
        train_index, test_index = next(sss.split(data, y))
        print("TRAIN:", train_index, "TEST:", test_index)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))

        train_df2 = data.iloc[train_index]
        test_df2 = data.iloc[test_index]
        print(train_df2.shape)
        print(test_df2.shape)

        temp_train_df2 = train_df2.groupby('label').count()
        print(temp_train_df2)
        temp_test_df2 = test_df2.groupby('label').count()
        print(temp_test_df2)

        print(train_df1.equals(train_df2))
        print(test_df1.equals(test_df2))

    def test_shuffle_split_array(self):
        X = [[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]]
        y = [0, 0, 0, 1, 1, 1]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_index, test_index = next(sss.split(X, y))
        print("TRAIN:", train_index, "TEST:", test_index)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))

    def test_monolingual_sentence_split(self):
        data_file_path = os.path.join(classifier_config.DATA_DIRECTORY,
                                      'subtask2-sentence/filtered/farm_format/en-train.tsv')
        config = classifier_config.config
        fold_id = 1

        # test reproducibility
        seed = int(config['manual_seed'] * (fold_id + 1))
        print(f"Set seed to {seed}.")
        train1, dev1 = split_sentence_data(data_file_path, seed, config, output_folder=None)
        train2, dev2 = split_sentence_data(data_file_path, seed, config, output_folder=None)
        self.assertEqual(True, train1.equals(train2))
        self.assertEqual(True, dev1.equals(dev2))

        # test randomness
        fold_id = 2
        seed = int(config['manual_seed'] * (fold_id + 1))
        print(f"Set seed to {seed}.")
        train3, dev3 = split_sentence_data(data_file_path, seed, config, output_folder=None)
        self.assertEqual(False, train1.equals(train3))
        self.assertEqual(False, dev1.equals(dev3))

    def test_monolingual_token_split(self):
        data_file_path = os.path.join(classifier_config.DATA_DIRECTORY,
                                      'subtask4-token/filtered/farm_format/en-train.txt')
        config = ner_config.config
        fold_id = 1

        # test reproducibility
        seed = int(config['manual_seed'] * (fold_id + 1))
        print(f"Set seed to {seed}.")
        train1, dev1 = split_token_data(data_file_path, seed, config, output_folder=None)
        train2, dev2 = split_token_data(data_file_path, seed, config, output_folder=None)
        # # to check similarity of the elements regardless of the order, arrays need to be sorted
        # train1_text = [x['text'] for x in train1]
        # print(train1_text.sort())
        self.assertEqual(True, train1 == train2)
        self.assertEqual(True, dev1 == dev2)

        # test randomness
        fold_id = 2
        seed = int(config['manual_seed'] * (fold_id + 1))
        print(f"Set seed to {seed}.")
        train3, dev3 = split_token_data(data_file_path, seed, config, output_folder=None)
        self.assertEqual(False, train1 == train3)
        self.assertEqual(False, dev1 == dev3)

    def test_multilingual_sentence_split(self):
        languages = ["en", "es", "pr"]
        data_dir = os.path.join(classifier_config.DATA_DIRECTORY, 'subtask2-sentence/filtered/farm_format')
        config = classifier_config.config
        fold_id = 1

        # # Check consistency between monolingual split and multilingual split per language
        seed = int(config['manual_seed'] * (fold_id + 1))
        print(f"Set seed to {seed}.")
        lang_splits = dict()
        for lang in languages:
            file_path = os.path.join(data_dir, f"{lang}-train.tsv")
            temp_train, temp_dev = split_sentence_data(file_path, seed, config, output_folder=None)
            lang_splits[lang] = [temp_train, temp_dev]

        train, dev = prepare_multilingual_sentence_data(languages, data_dir, seed, config, output_folder=None)

        for lang in languages:
            # each language split should be found in final split
            x = lang_splits[lang][0]
            # df_a.merge(df_b) returns intersection between df_a and df_b
            # If all the rows in df_a are available in df_b, merge will return same rows as df_a.
            bool_t = len(lang_splits[lang][0].merge(train).drop_duplicates()) == len(
                lang_splits[lang][0].drop_duplicates())
            print(f'{lang} monolingual train split found in final train split? {bool_t}')
            self.assertEqual(True, bool_t)

            bool_d = len(lang_splits[lang][1].merge(dev).drop_duplicates()) == len(
                lang_splits[lang][1].drop_duplicates())
            print(f'{lang} monolingual dev split found in final dev split? {bool_d}')
            self.assertEqual(True, bool_d)
