import torch
from spacy.lang.bn import Bengali
from data import config


def read_vocab(file_path="vocab.txt"):
    """
    Args:
        file_path: Saved vocabulary file path

    Returns: word_to_idx. A dictionary build on vocabulary with word to id mapping.
    """
    word_to_idx = dict()

    with open(file_path) as v:
        lines = v.readlines()

    for line in lines:
        word_with_id = line.rstrip().split(" : ")
        try:
            word_to_idx[word_with_id[0]] = int(word_with_id[1])
        except:
            continue
    return word_to_idx


def read_tags(file_path="tag.txt"):
    """
    Args:
        file_path: Saved text file path of available tags in the dataset

    Returns: id_to_tag. A dictionary build on saved tags with id to tag mapping.

    """
    id_to_tag = dict()

    with open(file_path) as v:
        lines = v.readlines()

    for line in lines:
        id_with_tag = line.rstrip().split(":")
        try:
            id_to_tag[int(id_with_tag[0])] = id_with_tag[1]
        except:
            continue

    return id_to_tag


def remove_punctuations(tokens):
    """
    Remove punctuations from the given sentence tokens
    Args:
        tokens: list of tokens

    Returns: tokens removing the punctuations

    """
    punctuations = r'''````£|¢|Ñ+-*/=EROero৳–•!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰🤣⚽️✌�￰'''

    for i, token in enumerate(tokens):
        m_token = ""
        for char in token:
            if char not in punctuations:
                m_token = m_token + char
        if len(m_token) == 0:
            tokens.remove(token)
        else:
            tokens[i] = m_token
    return tokens


def extract_names(tokens, tags):
    """
    Args:
        tokens: List of tokens of the input sentence
        tags: List of predicted tags from the model

    Returns:
        names: List of extracted names
    """
    name = ""
    b_tag = False

    for i, tag in enumerate(tags):
        if tag == 'B-PER':
            if b_tag:
                name = name + ":" + tokens[i]
            else:
                name = name + tokens[i]
                b_tag = True
        elif tag == "I-PER":
            name = name + " " + tokens[i]
    names = name.split(":")
    return names


def infer(checkpoint_path, sentence):
    """
    Args:
        checkpoint_path: Saved model checkpoint path.
        sentence: Input sentence on which inference will be done.

    Returns:
        tokens: List of generated tokens from given sentence.
        predicted_tags: Tags that are predicted with the trained model.
        unks: List of unknown tokens encountered in the sentence.
        names: List of extracted names from the sentence.
    """
    model = torch.jit.load(checkpoint_path)
    model.eval()
    # tokenize sentence
    nlp = Bengali()
    tokens = [token.text for token in nlp(sentence)]
    tokens = remove_punctuations(tokens)
    vocab_stoi = read_vocab()
    tags = read_tags()

    numerical_tokens = [vocab_stoi.get(t, 0) for t in tokens]
    # find unknown words
    unk_id = 0
    unks = [t for t, n in zip(tokens, numerical_tokens) if n == unk_id]

    token_tensor = torch.LongTensor(numerical_tokens)
    token_tensor = token_tensor.unsqueeze(-1)
    predictions = model(token_tensor)
    top_predictions = predictions.argmax(-1)

    predicted_tags = [tags[t.item()] for t in top_predictions]

    # {0: '<pad>', 1: 'O', 2: 'B-PER', 3: 'I-PER', 4: 'TIM', 5: 'B-ORG', 6: 'B-LOC', 7: 'I-ORG', 8: 'I-LOC'}
    names = extract_names(tokens, predicted_tags)

    # print("word".ljust(20), "entity")
    # print("-".ljust(30, "-"))
    #
    # for word, tag in zip(tokens, predicted_tags):
    #     print(word.ljust(20), tag)

    return tokens, predicted_tags, unks, names


def main():
    # checkpoint path of the trained model
    checkpoint_path = config.CHECKPOINT3
    sentences = ["মুনীর চৌধুরী এবং তানভীর হোসেন ছোটবেলার বন্ধু।", "সাইদুল সাহেব কাস্টমারকে একশ টাকা বাকি দিলেন।",
                 "আজ রাতে কোন রূপকথা নয়!"]

    for sen in sentences:
        words, infer_tags, unknown_tokens, names = infer(checkpoint_path, sen)
        print("Extracted names:", names)


if __name__ == '__main__':
    main()
