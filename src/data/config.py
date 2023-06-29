import os

DATA_BASE_PATH = "dataset"

FILE_NER1 = os.path.sep.join([DATA_BASE_PATH, "train_data.txt"])
FILE_NER2 = os.path.sep.join([DATA_BASE_PATH, "test_data.txt"])
NEW_FILE_NER1 = os.path.sep.join([DATA_BASE_PATH, "train_data_ner.csv"])
NEW_FILE_NER2 = os.path.sep.join([DATA_BASE_PATH, "test_data_ner.csv"])
UP_FILE_NER1 = os.path.sep.join([DATA_BASE_PATH, "train_up_ner.tsv"])
UP_FILE_NER2 = os.path.sep.join([DATA_BASE_PATH, "test_up_ner.tsv"])


CHECKPOINT_PATH = "checkpoint"
CHECKPOINT1 = os.path.sep.join([CHECKPOINT_PATH, "checkpoint_saved.pth"])
CHECKPOINT2 = os.path.sep.join([CHECKPOINT_PATH, "checkpoint.pth"])
CHECKPOINT3 = os.path.sep.join([CHECKPOINT_PATH, "model_scripted.pt"])
CHECKPOINT4 = os.path.sep.join([CHECKPOINT_PATH, "model_last.pt"])

