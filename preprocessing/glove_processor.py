from torch.nn.utils.rnn import pad_sequence

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

        self._create_lookup_tables()

        self._path = 'pre_trained\glove.6B.50d.txt'

        return None

    def _custom_collate_fn(self):

        # List of sentences -> List of tokens list

        # Get original length for pack padded sequence

        # List of tokens list -> List of integers list

        # List of integers list -> List of vectorized tokens

        # For padding process

        # For labels
        ...

    def _tokenizer(self, utterance: str):
        return [word.lower() for word in utterance.split(" ") if word != '']

    def _create_lookup_tables(self) -> None:

        """
            Create the following lookup tables:
            - Word-to-index: Word corresponds to specific index
            - Index-to-Vector: Index corresponds to specific vectors
        """

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

        # Save lookup table and word-integer transformation
        self.save_file()
            
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

    def _tokenize_utterances(self, utterances: list) -> list:
        """
            Convert list of sentences into list of words via tokenizer
        """

        tokenized_utterances = []

        for utterance in utterances:

            tokenized_utterances += [self._tokenizer(utterance)]

        return tokenized_utterances

    def begin_transformation(self, utterances_list):

        # List of sentences -> List of tokens list
        tokenized_list = self._tokenize_utterances(utterances_list)

        # List of tokens list -> List of integers list
        transformed_list = self._convert_text_integer(tokenized_list)

        # List of integers list -> List of vectorized tokens
        vectorized_list = self._convert_integers_vectors(transformed_list)

        return vectorized_list

    def save_file(self, file, filename):

        file = open(filename, 'wb')
        pickle.dump(file, filename)
        file.close()

        return None

    def load_file(self, filename):

        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()

        return data