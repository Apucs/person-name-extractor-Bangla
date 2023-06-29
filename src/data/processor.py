import pandas as pd
import csv
import os


def formatting(filename):
    """
    Args:
        filename: Name of the file which needs to be formatted.

    Returns: None
    """
    with open(filename, "r", encoding="utf8") as in_file:
        buf = in_file.readlines()

    with open(filename, "w", encoding="utf8") as out_file:
        for line in buf:
            string = r"।"
            if string in line:
                line = line + "\n"
            out_file.write(line)


def csv_to_tsv(up_file_name):
    """
    Convert the CSV file into TSV file
    Args:
        up_file_name: Updated file name

    Returns: None. Format and save the file into TSV format
    """
    new_file_name = os.path.splitext(up_file_name)[0] + ".tsv"

    with open(up_file_name, 'r', encoding='utf-8') as csvin, open(new_file_name, 'w', newline='',
                                                                  encoding='utf-8') as tsvout:
        csvin = csv.reader(csvin)
        next(csvin, None)
        tsvout = csv.writer(tsvout, delimiter='\t')

        for row in csvin:
            tsvout.writerow(row)

    formatting(new_file_name)


def process_label(df):
    """
    Process and modify the label if needed.
    Args:
        df: dataframe from which label will be modified.

    Returns: None

    """
    if df['O_BI'] == "B-PER":
        return "B-PER"
    elif df['O_BI'] == "I-PER":
        return "I-PER"
    else:
        return 'O'


def remove_punctuations(df):
    """
    Remove punctuations from the given dataframe
    Args:
        df: dataframe

    Returns: Dataframe removing punctuations

    """
    punctuations = r'''````£|¢|Ñ+-*/=EROero৳–•!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰🤣⚽️✌�￰'''

    data = df.copy()
    for word in data['word']:
        m_word = ""
        for char in word:
            if char not in punctuations:
                m_word = m_word + char
        if len(m_word) == 0:
            data.drop(data[data['word'] == word].index, inplace=True)
        else:
            data.replace(word, m_word, inplace=True)
    return data


def remove_stopwords(df):
    """
    Remove stopwords from the given dataframe
    Args:
        df: dataframe

    Returns: Dataframe removing stopwords

    """
    data1 = pd.read_csv('stopwords_bangla.csv')
    stop_words = data1['words'].tolist()

    data = df.copy()
    for word in data['word']:
        if word in stop_words:
            data.drop(data[data['word'] == word].index, inplace=True)

    return data


def process_data(file_path, file_name, up_file_name=None):
    """
    Process the dataset and save the processed dataset into desired format.
    Args:
        file_path: Base file path which will be processed.
        file_name: Name of the file in which it will be saved
        up_file_name: Updated file name after modifying the label(If needed)

    Returns: none

    """

    df = pd.read_csv(file_path, sep='\s+', names=['word', 'O_BI'], header=None)
    df = df.fillna(method="ffill")
    df = remove_punctuations(df)
    # df = remove_stopwords(df)  # If stopwords need to be removed
    df.to_csv(file_name, index=False)
    csv_to_tsv(file_name)

    # Uncomment the following block if only labels need to modify.

    # if up_file_name is not None:
    #     df = pd.read_csv(file_name)
    #     df['O_BI'] = df.apply(process_label, axis=1)
    #     df_updated = df.copy()
    #     df_updated.to_csv(up_file_name, index=False)
    #     csv_to_tsv(up_file_name)
