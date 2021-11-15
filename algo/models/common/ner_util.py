# Created by Hansi at 11/15/2021
import re


def to_iob(sentences, predictions):
    """
    Convert raw NER output to IOB2 format
    :param sentences: list of dict {'text': "sample text"}
    :param predictions: list of dict {'start': i, 'end': j, 'context':"sample text", 'label': 'predicted label', 'probability': 0.9404173}
    :return: list
        list of list which contains IOB tags
    """
    iob_outputs = []
    for idx, sample_labels in enumerate(predictions):
        indices = [(ele.start(), ele.end()) for ele in re.finditer(r'\S+', sentences[idx]["text"])]
        iob_output = ["O" for index in indices]
        for label in sample_labels:
            if label['label'] in ["[PAD]", "X"]:
                continue
            for i, ind in enumerate(indices):
                if ind[0] == label['start']:
                    iob_output[i] = f"B-{label['label']}"
                    if ind[1] != label['end']:
                        end_word_index = 0
                        for j in range(i + 1, len(indices)):
                            if indices[j][1] == label['end']:
                                end_word_index = j
                                break
                        if end_word_index == 0:
                            raise ValueError(f"Span index error found in output: {label}")
                        for i_index in range(i + 1, end_word_index + 1):
                            iob_output[i_index] = f"I-{label['label']}"
                    break
        iob_outputs.append(iob_output)
    return iob_outputs


def to_binary(sentence, predictions):
    binary_outputs = []
    for idx, sample_labels in enumerate(predictions):
        binary_output = []
        for label in sample_labels:
            label_val = label['label']
            binary_output.append(0 if label_val in ["[PAD]", "X"] else int(label_val))
        binary_outputs.append(binary_output)
    return binary_outputs
