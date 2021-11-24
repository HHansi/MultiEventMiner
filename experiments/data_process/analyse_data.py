# Created by Hansi at 9/20/2021
import os

import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize

from experiments.data_process.data_util import read_data_df
from experiments.data_config import DATA_DIRECTORY


def plot_seq_length_histo(texts, plot_path=None, type='sentence'):
    if type == 'sentence':
        seq_lengths = [len(word_tokenize(x)) for x in texts]
    else:  # each element in texts is a list of tokens
        seq_lengths = [len(x) for x in texts]
    print(seq_lengths)

    print(f'Min: {min(seq_lengths)}')
    print(f'Max: {max(seq_lengths)}')

    binwidth = 10
    density, bins, bars = plt.hist(seq_lengths, bins=range(0, max(seq_lengths) + binwidth, binwidth), rwidth=0.5,
                                   color='#607c8e')
    counts, _ = np.histogram(seq_lengths, bins)
    print(counts)
    print(bins)

    plt.xlabel('Sequence Length')
    plt.ylabel('Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(np.arange(0, max(seq_lengths) + binwidth, binwidth * 4))
    if plot_path is not None:
        plt.savefig(plot_path)
    plt.show()


def get_class_distribution(data_path):
    df = read_data_df(data_path)
    # temp_df = df[['sentence', 'label']]
    temp_df = df.groupby('label').count()
    print(temp_df)
    # temp_df.plot.bar()


if __name__ == '__main__':
    data_path = os.path.join(DATA_DIRECTORY, '../data/subtask2-sentence/filtered/es-train.json')
    # data_path = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/es-test.json')
    # df = read_data_df(data_path)
    # texts = df['sentence'].tolist()

    # data_path = os.path.join(DATA_DIRECTORY, 'subtask4-token/filtered/es-en-en-train.txt')
    # tokens, labels = read_tokens(data_path, train=True)
    # # labels = [[] for i in tokens]
    # token_data, sentence_tokens, sentence_labels, token_sentence_count, sentence_without_trigger_count = get_token_sentences(
    #     tokens, labels)
    # texts = sentence_tokens

    # plot_path = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/plots/es-test.png')
    # plot_seq_length_histo(texts, plot_path=plot_path, type='sentence')

    get_class_distribution(data_path)
