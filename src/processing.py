from data import processor
from data import config

"""
This script will process the dataset and save them in two formats. 1. CSV format, 2. TSV format
"""

file_names = [config.FILE_NER1, config.FILE_NER2]

new_file_names = [config.NEW_FILE_NER1, config.NEW_FILE_NER2]

up_file_names = [config.UP_FILE_NER1, config.UP_FILE_NER2]

for i, (fn, nfn, ufn) in enumerate(zip(file_names, new_file_names, up_file_names)):
    print(i, "\t", fn, "\t", nfn, "\t", ufn)
    processor.process_data(fn, nfn, ufn)






