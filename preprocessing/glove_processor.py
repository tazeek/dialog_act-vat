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
    """

    def _load_model(self):

        self._path = 'pre_trained\glove.6B.50d.txt'
        self._glove_weights = self._load_glove_model()

        self._word2idx = self._create_word_index()

        return None

    def _tokenizer(self, utterance: str):
        return [word.lower() for word in utterance.split(" ") if word != '']

    def _convert_word_index(self, words_list: list) -> dict:

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

    def _preprocess_text(self, utterances: list) -> list:

        processed_utterances = []

        for utterance in utterances:

            processed_utterances += [self._tokenizer(utterance)]


        return processed_utterances

    def _load_glove_model(self):

        glove = {}

        with open(self._path, encoding="utf8") as f:

            for line in f.readlines():

                values = line.split()
                
                # Index 0: Word
                # Index 1 onwards: Weights
                word = values[0]

                glove[word] = np.array(values[1:], dtype='float32')
            
        return glove

    def _create_word_index(self):

        # Load the full data of DailyDialog
        label_loader = dailydialog_full.DailyDialog_Full().fetch_dataframe()

        # Preprocess the text and get tokens for dictionary creation
        all_possible_words_list = self._preprocess_text(
            label_loader['utterance']
        )

        # Get word to index dictionary
        return self._convert_word_index(all_possible_words_list)

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

    def create_embeddings(self):

        embeddings = np.zeros((len(self._word2idx), 50))

        count = 0
        # Loop through word_index_dict
        for word, index in self._word2idx.items():

            # Check if item is in glove_vectors
            if word in self._glove_weights:
                count += 1
                embeddings[index] = self._glove_weights[word]

        return torch.from_numpy(embeddings).float()

    def transform_data(self, words_list):

        transformed_list = []

        for words in words_list:
            transformed_list += [[self._word2idx[word] for word in words]]
        
        return transformed_list

    def embeddings_test():

        ...

    def save_transformed_data():

        ...