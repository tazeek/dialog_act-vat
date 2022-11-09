from string import punctuation
from torchtext.data import get_tokenizer
from torch.autograd import Variable

import torch
import numpy as np

def preprocess_text(utterances: list, remove_punctuation:'bool'=True) -> list:

    processed_utterances = []
    tokenizer = get_tokenizer("basic_english")

    """
        TODO:
        - Complete the sentence (example: i'll -> i will)
        - Need to lowercase?
    """

    for utterance in utterances:
        # Split based on tokenizer
        words = tokenizer(utterance)

        # Remove punctuation (OPTIONAL)
        if remove_punctuation:
            words = [word for word in words if word not in punctuation]

        processed_utterances += [words]


    return processed_utterances

def convert_word_index(words_list: list) -> dict:

    word_to_index = {}
    index = 0

    
    for list in words_list:

        for word in list:

            # Check if word does not exist
            # and Increment index
            if word not in word_to_index:
                word_to_index[word] = index
                index += 1

    return word_to_index

def load_glove_model():

    path = 'pre_trained\glove.6B.50d.txt'
    glove = {}

    with open(path, encoding="utf8") as f:

        for line in f.readlines():
            values = line.split()
            
            word = values[0]
            vector = np.array(values[1:], dtype='float32')

            glove[word] = vector
        
    return glove

def create_glove_embeddings(glove_vectors: dict, word_index_dict: dict) -> list:

    embeddings = np.zeros((len(word_index_dict), 50))
    count = 0
    # Loop through word_index_dict
    for word, index in word_index_dict.items():

        # Check if item is in glove_vectors
        if word in glove_vectors:
            count += 1
            embeddings[index] = glove_vectors[word]

    print(f"Number of words found: {count}")

    return torch.from_numpy(embeddings).float()

def embeddings_test(embeddings: dict, glove_vectors: dict, word_index_dict: dict, test_str: str):

    # Get the index first
    index = word_index_dict[test_str]

    # Print out embeddings first
    embedding_test = embeddings(Variable(torch.LongTensor([index])))
    print(embedding_test)

    print('\n\n')
    # Get the glove value
    glove_test = glove_vectors[test_str]
    print(glove_test)
    
    return None