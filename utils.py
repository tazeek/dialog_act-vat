from string import punctuation
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import numpy as np

def _tokenizer(utterance: str):
    return [word.lower() for word in utterance.split(" ") if word != '']

def _custom_collate(data: list):

    # Get original input length for pack padded sequence
    input_len = [len(d['text']) for d in data]
    input_len = torch.tensor(input_len)

    # For padding process
    inputs = [torch.tensor(d['text']) for d in data]
    inputs_padded = pad_sequence(inputs, batch_first=True)

    # For labels
    labels = [d['class'] for d in data]
    labels = torch.tensor(labels)

    return input_len, inputs_padded, labels

def preprocess_text(utterances: list, remove_punctuation:'bool'=True) -> list:

    processed_utterances = []

    """
        TODO:
        - Complete the sentence (example: i'll -> i will)
        - Need to lowercase?
    """

    for utterance in utterances:
        # Split based on tokenizer
        words = _tokenizer(utterance)

        # Remove punctuation (OPTIONAL)
        if remove_punctuation:
            words = [word for word in words if word not in punctuation]

        processed_utterances += [words]


    return processed_utterances

def transform_text_integer(words_list: list, index_dict: dict) -> list:

    transformed_list = []

    for words in words_list:
        transformed_list += [[index_dict[word] for word in words]]
    
    return transformed_list

def convert_word_index(words_list: list) -> dict:

    word_to_index = {0: '<pad>'}
    index = 1
    
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

def transform_dataloader(dataloader_dataset):
    return DataLoader(dataloader_dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=_custom_collate
    )

def train_model(dataloader):

    # Loop through dataloader
    for (length, padded_input, label) in dataloader:

        # Transform the data based on the lookup table
        print(length)
        print(padded_input)
        print(label)
        break
    
    exit()
