# Created by Hansi at 9/16/2021
import os

from nltk import word_tokenize
from tqdm import tqdm

from experiments.data_config import DATA_DIRECTORY
from experiments.data_process.data_util import read_data_df, save_data, read_tokens, save_tokens, \
    read_tokens_farm_format


# detokenizer = TreebankWordDetokenizer()

class SentenceToken:
    def __init__(self, sentence_tokens, sentence_labels):
        # self.sentence = sentence
        self.sentence_tokens = sentence_tokens
        self.sentence_labels = sentence_labels


def drop_sentence_level_duplicates(input_path, output_path):
    df = read_data_df(input_path)
    print(f'Original count: {df.shape[0]}')
    # df = df.drop_duplicates()
    df.drop_duplicates(subset=['sentence'], inplace=True)
    print(f'Without duplicates count: {df.shape[0]}')

    data = df.to_dict(orient='records')
    save_data(data, output_path)


def drop_token_level_duplicates(input_path, output_path):
    tokens, labels = read_tokens(input_path)
    print(f'Original count: {len(tokens)}')
    duplicate_indices = set()

    for i in range(len(tokens)):
        temp_token = tokens[i]
        for j in range(i + 1, len(tokens)):
            if tokens[j] == temp_token:
                print(f'j:{j}, i:{i}')
                duplicate_indices.add(j)

    for index in duplicate_indices:
        del tokens[index]
        del labels[index]
    print(f'Without duplicates count: {len(tokens)}')

    save_tokens(tokens, labels, output_path)


def get_token_sentences(tokens, labels):
    sentence_count = 0
    sentence_without_trigger_count = 0
    token_data = []
    sentence_tokens = []
    sentence_labels = []
    dict_instance_sentence = dict()
    instance_index = -1
    sentence_index = -1
    for temp_tokens, temp_labels in zip(tokens, labels):
        instance_index += 1
        instance_sentence_indices = []

        SEP_indices = [i for i, value in enumerate(temp_tokens) if value == '[SEP]']

        if len(SEP_indices) == 0:  # If no [SEP] labels found, the instance has one sentence.
            sentence_token = SentenceToken(temp_tokens, temp_labels)
            sentence_count += 1
            sentence_tokens.append(temp_tokens)
            sentence_labels.append(temp_labels)
            token_data.append([temp_tokens, temp_labels, [sentence_token]])
        else:
            # token_sentence_count += len(SEP_indices)
            SEP_indices.insert(0, -1)  # Add the index of -1 to the beginning
            SEP_indices.insert(len(SEP_indices), len(temp_tokens))

            sentence_token_list = []  # list of SentenceToken objects
            for i in range(0, len(SEP_indices)):
                if i < (len(SEP_indices) - 1):
                    temp_sent_tokens = temp_tokens[SEP_indices[i] + 1:SEP_indices[i + 1]]
                    # sentence = detokenizer.detokenize(sentence_tokens)

                    temp_sent_labels = []
                    if len(temp_labels) > 0:
                        temp_sent_labels = temp_labels[SEP_indices[i] + 1:SEP_indices[i + 1]]
                        if "B-trigger" not in temp_sent_labels:
                            sentence_without_trigger_count += 1

                    sentence_token = SentenceToken(temp_sent_tokens, temp_sent_labels)
                    sentence_count += 1
                    sentence_tokens.append(temp_sent_tokens)
                    sentence_labels.append(temp_sent_labels)

                    sentence_token_list.append(sentence_token)
            token_data.append([temp_tokens, temp_labels, sentence_token_list])
    return token_data, sentence_tokens, sentence_labels, sentence_count, sentence_without_trigger_count


def filter_shared_instances(train_path, test_path, train_level, test_level, output_folder, farm_format=False):
    # Token-level data reading and processing
    if train_level == 'token':
        tokens, labels = read_tokens(train_path)
    elif test_level == 'token':
        if farm_format:  # ToDO - only implemented for test token level
            tokens, labels = read_tokens_farm_format(test_path)
        else:
            tokens, labels = read_tokens(test_path, train=False)
            labels = [[] for i in tokens]

    else:
        raise ValueError("No token level found!")

    token_data, sentence_tokens, sentence_labels, token_sentence_count, sentence_without_trigger_count = get_token_sentences(
        tokens, labels)

    if len(token_data) == 0:
        raise ValueError("No token data found!")
    temp_count = 0
    for element in token_data:
        temp_count += len(element[2])
    print(temp_count)

    print(
        f'Loaded token data: instances={len(token_data)}, sentences={token_sentence_count}, '
        f'sentences without triggers={sentence_without_trigger_count}')

    # Sentence-level data reading
    if train_level == 'sentence':
        df = read_data_df(train_path)
    elif test_level == 'sentence':
        df = read_data_df(test_path)
    else:
        raise ValueError("No sentence level found!")
    if df.empty:
        raise ValueError("No sentence data found!")

    print(f'Loaded sentence data: instances={df.shape[0]}')

    # Filtering - if test instance is found in train, remove it from train
    if test_level == 'token':
        matched_df_indices = []
        for test_instance in tqdm(token_data):
            for sentence_token in test_instance[2]:
                for index, row in df.iterrows():
                    if word_tokenize(row['sentence']) == sentence_token.sentence_tokens:
                        print(f'duplicate: {sentence_token.sentence_tokens}')
                        matched_df_indices.append(index)
                        break  # consider data with no duplicate sentences

        df = df.drop(matched_df_indices)
        print(f'Removed instance count: {len(matched_df_indices)}')

    elif test_level == 'sentence':
        removed_n = 0
        for index, row in tqdm(df.iterrows()):
            for train_instance in token_data:
                # updated_sentence_token_list = [x for x in train_instance[2] if x.sentence_tokens != word_tokenize(row['sentence'])]
                updated_sentence_token_list = []
                for x in train_instance[2]:
                    if x.sentence_tokens != word_tokenize(row['sentence']):
                        updated_sentence_token_list.append(x)
                    else:
                        print(f"duplicate: {row['sentence']}")

                removed_n += len(train_instance[2]) - len(updated_sentence_token_list)
                # updated sentence list can be empty.
                train_instance[2] = updated_sentence_token_list
        print(f'Removed instance count: {removed_n}')

    final_token_sentence_count = 0
    final_sentence_without_trigger_count = 0

    # Save token data - sentences within the same instance are merged with [SEP]
    # Only training data should saved, because testing data did not update.
    if train_level == 'token':
        final_tokens = []
        final_labels = []
        for instance in token_data:
            merged_tokens = []
            merged_labels = []
            final_token_sentence_count += len(instance[2])
            for i in range(len(instance[2])):
                merged_tokens.extend(instance[2][i].sentence_tokens)
                if len(instance[2][i].sentence_labels) > 0:
                    merged_labels.extend(instance[2][i].sentence_labels)
                    if "B-trigger" not in instance[2][i].sentence_labels:
                        final_sentence_without_trigger_count += 1
                if i != len(instance[2]) - 1:
                    merged_tokens.extend(['[SEP]'])
                    if len(instance[2][i].sentence_labels) > 0:
                        merged_labels.extend(['O'])

            if len(merged_tokens) > 0:
                final_tokens.append(merged_tokens)
                if len(merged_labels) > 0:
                    final_labels.append(merged_labels)
        save_tokens(final_tokens, final_labels, os.path.join(output_folder, os.path.basename(train_path)))
        print(f'Saved token data: instances={len(final_tokens)}, sentences={final_token_sentence_count}, '
              f'sentences without triggers={final_sentence_without_trigger_count}')

    # Save sentence data
    if train_level == 'sentence':
        data = df.to_dict(orient='records')
        save_data(data, os.path.join(output_folder, os.path.basename(train_path)))
        print(f'Saved sentence data: instances={df.shape[0]}')


def get_token_level_duplicates(tokens, sent_df):
    """

    :param tokens: list
        list of tokens - [[t1, t2, t3], [t1, t2, ..]..]
    :param sent_df: dataframe
        dataframe with column 'sentence'
    :return: list
        indices of tokens which are found in sentences
    """
    duplicate_ids = []
    sentences = sent_df['sentence'].tolist()
    sentence_tokens = [word_tokenize(sent) for sent in sentences]

    for idx, token in tqdm(enumerate(tokens)):
        if token in sentence_tokens:
            print(f'duplicate: {token}')
            duplicate_ids.append(idx)

    # for idx, token in tqdm(enumerate(tokens)):
    #     for index, row in sent_df.iterrows():
    #         if word_tokenize(row['sentence']) == token:
    #             print(f'duplicate: {token}')
    #             duplicate_ids.append(idx)
    #             break
    return duplicate_ids


def count_common_docs_to_sentence_level(doc_file_path, sentence_file_path):
    df_doc = read_data_df(doc_file_path)
    df_sent = read_data_df(sentence_file_path)
    common_ids = []  # sent_id-doc_id
    common_doc_ids = []
    common_sent_ids = []

    for index_sent, row_sent in tqdm(df_sent.iterrows()):
        for index_doc, row_doc in df_doc.iterrows():
            if row_sent["sentence"] in row_doc["text"]:
                common_ids.append(f"{row_sent['id']}-{row_doc['id']}")
                common_doc_ids.append(f"{row_doc['id']}")
                common_sent_ids.append(f"{row_sent['id']}")
                break
    print(f"Found {len(common_ids)} common instances!")
    print(f"Common doc count = {len(common_doc_ids)}")
    print(f"Common sent count = {len(common_sent_ids)}")
    print(common_ids)


if __name__ == '__main__':
    # train_path = os.path.join(DATA_DIRECTORY, 'subtask4-token/es-train.txt')
    # test_path = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/es-test.json')
    # test_level = 'sentence'
    # train_level = 'token'
    # output_folder = os.path.join(DATA_DIRECTORY, 'subtask4-token/filtered2')

    # train_path = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/es-train.json')
    # train_path = os.path.join(DATA_DIRECTORY, '../data/subtask2-sentence/without_duplicates/es-train.json')
    # test_path = os.path.join(DATA_DIRECTORY, '../data/subtask4-token/es-test.txt')
    # test_level = 'token'
    # train_level = 'sentence'
    # output_folder = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/filtered2')

    train_path = os.path.join(DATA_DIRECTORY, '../data/subtask2-sentence/filtered/en-train.json')
    test_path = os.path.join(DATA_DIRECTORY, '../data/subtask4-token/filtered/farm_format/split_binary/en-dev.txt')
    test_level = 'token'
    train_level = 'sentence'
    output_folder = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/filtered2')

    # filter_shared_instances(train_path, test_path, train_level, test_level, output_folder, farm_format=True)

    # output_path = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/without_duplicates/es-train.json')
    # drop_sentence_level_duplicates(train_path, output_path)

    # output_path = os.path.join(DATA_DIRECTORY, 'subtask4-token/without_duplicates/es-en-en-train.txt')
    # drop_token_level_duplicates(train_path, output_path)

    # # count common instances
    # Sentence test -> Doc test
    # doc_file_path = os.path.join(DATA_DIRECTORY, 'subtask1-doc/en-test.json')
    # sentence_file_path = os.path.join(DATA_DIRECTORY, 'subtask2-sentence/en-test.json')
    # count_common_docs_to_sentence_level(doc_file_path, sentence_file_path)
