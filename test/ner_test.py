# Created by Hansi at 10/4/2021

import unittest

from algo.models.ner_model import NERModel
from experiments.ner_experiment import majority_class_for_ner


class TestNERFormatting(unittest.TestCase):
    def test_toIOB(self):
        test_sentences = [
            {"text": "Paris is a town in France"},
            {"text": "Only France and Britain backed proposal ."},
            {"text": "Peter Blackburn"},
        ]
        test_predictions =[[{'start': 0, 'end': 5, 'context': 'Paris', 'label': 'X', 'probability': 0.23707348}],
                           [{'start': 5, 'end': 11, 'context': 'France', 'label': 'LOC', 'probability': 0.95319456}, {'start': 16, 'end': 23, 'context': 'Britain', 'label': 'LOC', 'probability': 0.97308844}],
                           [{'start': 0, 'end': 15, 'context': 'Peter Blackburn', 'label': 'PER', 'probability': 0.30246672}]]

        expected_output = [['O', 'O', 'O', 'O', 'O', 'O'],
                           ['O', 'B-LOC', 'O', 'B-LOC', 'O', 'O', 'O'],
                           ['B-PER', 'I-PER']]

        output = NERModel.to_iob(test_sentences, test_predictions)
        print(output)
        self.assertEqual(output, expected_output)


class TestNERPredictions(unittest.TestCase):
    def test_majority_class_for_ner(self):
        test_sentences = [
            {"text": "Paris is a town in France"},
            {"text": "Only France and Britain backed proposal ."},
            {"text": "Peter Blackburn"},
        ]

        n_folds = 3
        # fold 0 predictions
        fold_0_1 = ['O', 'O', 'O', 'O', 'O', 'O']
        fold_0_2 = ['O', 'B-LOC', 'O', 'B-LOC', 'O', 'O', 'O']
        fold_0_3 = ['B-PER', 'I-PER']
        fold_0_output = [fold_0_1, fold_0_2, fold_0_3]

        # fold 1 predictions
        fold_1_1 = ['B-LOC', 'O', 'O', 'O', 'O', 'O']
        fold_1_2 = ['O', 'B-LOC', 'O', 'B-LOC', 'O', 'O', 'O']
        fold_1_3 = ['B-PER', 'I-PER']
        fold_1_output = [fold_1_1, fold_1_2, fold_1_3]

        # fold 2 predictions
        fold_2_1 = ['B-LOC', 'O', 'O', 'O', 'O', 'O']
        fold_2_2 = ['O', 'O', 'O', 'O', 'O', 'O', 'O']
        fold_2_3 = ['B-PER', 'I-PER']
        fold_2_output = [fold_2_1, fold_2_2, fold_2_3]

        preds = [fold_0_output, fold_1_output, fold_2_output]  # [[fold_0 predictions], ... [fold_n predictions]]

        expected_output = [['B-LOC', 'O', 'O', 'O', 'O', 'O'],
                           ['O', 'B-LOC', 'O', 'B-LOC', 'O', 'O', 'O'],
                           ['B-PER', 'I-PER']]

        output = majority_class_for_ner(test_sentences, preds, n_folds=n_folds)
        print(output)
        self.assertEqual(output, expected_output)

