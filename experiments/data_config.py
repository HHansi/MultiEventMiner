# Created by Hansi at 9/16/2021
import os

SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_PATH, 'data')

SENTENCE_TRAIN_DATA_FILE = "train.tsv"
SENTENCE_DEV_DATA_FILE = "dev.tsv"

TOKEN_TRAIN_DATA_FILE = "train.txt"
TOKEN_DEV_DATA_FILE = "dev.txt"