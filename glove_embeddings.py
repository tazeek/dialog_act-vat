from torch import nn
from torch.autograd import Variable

import numpy as np
import torch

def _load_glove_model():

    path = 'pre_trained\glove.6B.50d.txt'
    glove = {}

    with open(path, encoding="utf8") as f:

        for line in f.readlines():
            values = line.split()
            
            word = values[0]
            vector = np.array(values[1:], dtype='float32')

            glove[word] = vector
        
    return glove

def _do_embeddings_test(embeddings: dict, glove_vectors: dict, word_index_dict: dict, test_str: str):

    # Create the embedding layer
    embedding_layer = nn.Embedding(embeddings.size(0), embeddings.size(1))
    embedding_layer.weight = nn.Parameter(embeddings)
    
    # Get the index first
    index = word_index_dict[test_str]

    # Print out embeddings first
    embedding_test = embedding_layer(Variable(torch.LongTensor([index])))
    print(embedding_test)

    print('\n\n')
    # Get the glove value
    glove_test = glove_vectors[test_str]
    print(glove_test)

    return None

def create_glove_embeddings(word_index_dict: dict, embeddings_test:'bool'=False) -> list:

    # Load glove model
    glove_vectors = _load_glove_model()
    print(f"Length of GloVe model: {len(glove_vectors)}")

    embeddings = np.zeros((len(word_index_dict), 50))
    count = 0
    # Loop through word_index_dict
    for word, index in word_index_dict.items():

        # Check if item is in glove_vectors
        if word in glove_vectors:
            count += 1
            embeddings[index] = glove_vectors[word]

    print(f"Number of words found: {count}")

    if embeddings_test:
        _do_embeddings_test(embeddings, glove_vectors, word_index_dict, 'cake')

    return torch.from_numpy(embeddings).float()