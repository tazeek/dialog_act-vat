from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data_loaders import custom_dataloader

import numpy as np
import torch
import pickle

class Glove_Processor:

    """
        Preprocessing Steps (from raw to transformed):
        
        1. Load the model weights
        2. Create the lookup table from dailydialog vocab
        3. Tokenization of utterances from dataset
        4. Convert tokens to integers
        5. Convert the integers to vector tokens via embedding
        6. Save the transformed vector tokens

        Batching for Deep Learning (from transformed to batch):

        1. Load the vector tokens, original length, and label
        2. Transform to data loader

        Final output: B * N * V
        B -> Batch Size (Number of utterances)
        N -> Number of tokens
        V -> Vector size of each token
        
        V is always 50
    """

    def __init__(self, logger):
        self._logger = logger

        self._path = 'pre_trained\glove.6B.50d.txt'

        self._word2idx = {'<pad>': 0}
        self._lookup_table_glove = {}

    def _tokenizer(self, utterance: str):
        return [word.lower() for word in utterance.split(" ") if word != '']

    def _create_lookup_tables(self) -> None:

        """
            Create the following lookup tables:
            - Word-to-index: Word corresponds to specific index
            - Index-to-Vector: Index corresponds to specific vectors
        """

        index = 1

        # For padded embeddings
        self._lookup_table_glove[0] = np.zeros(50)

        self._logger.info(f'Loading GloVe vectors and creating the word-index dictionary')

        with open(self._path, encoding="utf8") as f:

            for line in f.readlines():

                values = line.split()
                
                # Index 0: Word
                # Index 1 onwards: Weights
                word = values[0]
                weights_val = [float(vector) for vector in values[1:]]

                # Storage time
                self._lookup_table_glove[index] = torch.FloatTensor(weights_val)
                self._word2idx[word] = index

                index += 1

        # Save lookup table and word-integer transformation
        self._logger.info(f'Saving the lookup table and word-index dictionary')

        torch.save(self._lookup_table_glove, 'lookup_table_glove.pt')

        with open('word_to_index.pickle', 'wb') as file:
            pickle.dump(self._word2idx, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        return None

    def _convert_text_integer(self, words_list):

        """
            Convert list of words into list of integers
            by the use of a lookup table
        """

        transformed_list = []

        for words in words_list:
            transformed_list += [
                [
                    self._word2idx[word] for word in words
                    if word in self._word2idx
                ]
            ]
        
        return transformed_list

    def _convert_integers_vectors(self, index_list):

        """
            Convert list of integers into list of vectors
            by the use of a lookup table
        """

        vectorized_list = []

        for numbers_list in index_list:

            vectorized_list += [
                [
                    self._lookup_table_glove[index]
                    for index in numbers_list
                ]
            ]

        return vectorized_list

    def _tokenize_utterances(self, utterances: list) -> list:
        """
            Convert list of sentences into list of words via tokenizer
        """

        tokenized_utterances = []

        for utterance in utterances:

            tokenized_utterances += [self._tokenizer(utterance)]

        return tokenized_utterances

    def _custom_collate_fn(self, data: tuple):

        utterances, labels = zip(*data)

        # List of sentences -> List of tokens list
        tokenized_list = self._tokenize_utterances(utterances)

        # Get original length for pack padded sequence
        # Then, convert to tensor
        original_len = [len(tokens_list) for tokens_list in tokenized_list]
        original_len = torch.tensor(original_len)

        # List of tokens list -> List of integers list
        transformed_list = self._convert_text_integer(tokenized_list)

        # List of integers list -> List of vectorized tokens
        vectorized_list = self._convert_integers_vectors(transformed_list)

        # For padding process
        vector_inputs = [torch.tensor(vector_list) for vector_list in vectorized_list]
        vector_inputs_padded = pad_sequence(vector_inputs, batch_first=True)

        # For labels
        labels = [label for label in labels]
        labels = torch.tensor(labels)
        
        return {
            'original_length': original_len,
            'features': vector_inputs_padded,
            'labels': labels
        }

    def begin_transformation(self, text, labels, batch_size):

        # Load the models
        self._create_lookup_tables()
        self._logger.info('Table creation successful.')

        # Use customized dataset
        self._logger.info('Loading DataLoader....')
        dataset = custom_dataloader.CustomDataLoader(text, labels)

        return DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._custom_collate_fn
        )