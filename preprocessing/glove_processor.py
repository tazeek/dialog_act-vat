from data_loaders import dailydialog_full
from preprocessing import text_processing
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import torch

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

        1. Load the vector tokens
        2. Perform collate function
        3. Transform to data loader

        Final output: B * N * V
        B -> Batch Size (Number of utterances)
        N -> Number of tokens
        V -> Vector size of each token
        
        V is always 50
    """

    def _load_model(self):

        self._word2idx = {'<pad>': 0}
        self._lookup_table_glove = {}

        self._create_lookup_table()

        self._path = 'pre_trained\glove.6B.50d.txt'

        return None

    def _tokenizer(self, utterance: str):
        return [word.lower() for word in utterance.split(" ") if word != '']

    def _create_lookup_table(self) -> None:

        index = 1

        with open(self._path, encoding="utf8") as f:

            for line in f.readlines():

                values = line.split()
                
                # Index 0: Word
                # Index 1 onwards: Weights
                word = values[0]
                self._lookup_table_glove[index] = torch.from_numpy(values[1:]).float()

                self._word2idx[word] = index

                index += 1
            
        return None

    def _convert_text_integer(self, words_list):

        transformed_list = []

        for words in words_list:
            transformed_list += [[self._word2idx[word] for word in words]]
        
        return transformed_list

    def _custom_collate(self, data: list):

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

    def preprocess_text(self, utterances: list) -> list:

        tokenized_utterances = []

        for utterance in utterances:

            tokenized_utterances += [self._tokenizer(utterance)]

        return tokenized_utterances

    def begin_transformation(self, utterances_list):

        # List of sentences -> List of tokens list
        tokenized_list = self.preprocess_text(utterances_list)

        # List of tokens list -> List of integers list
        transformed_list = self._convert_text_integer(tokenized_list)

        # List of integers list -> List of vectorized tokens

        ...

    def embeddings_test():

        ...

    def save_transformed_data():

        ...