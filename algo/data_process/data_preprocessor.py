# Created by Hansi at 4/9/2021
import re

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '..', '...', '…']


def remove_links(sentence, substitute=''):
    """
    Method to remove links in the given text
    parameters
    -----------
    :param sentence: str
    :param substitute: str
        which to replace link
    :return: str
        String without links
    """
    sentence = re.sub('https?:\/\/\S+', substitute, sentence, flags=re.MULTILINE)
    return sentence.strip()


def remove_repeating_characters(sentence):
    """
    remove non alphaneumeric characters which repeat more than 3 times by its 3 occurrence (e.g. ----- to ---)
    :param sentence:
    :return:
    """
    sentence = re.sub('(\W)\\1{3,}', '\\1', sentence)
    return sentence.strip()


