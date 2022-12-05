from data_loaders import dailydialog_full
from preprocessing import text_processing

import numpy as np

class Glove_Processor:

    def __init__(self):

        self._path = 'pre_trained\glove.6B.50d.txt'
        self._weights = self._load_glove_model()

        self._word2idx = self._create_word_index()

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
        all_possible_words_list = text_processing.preprocess_text(
            label_loader['utterance'], 
            remove_punctuation=False
        )

        # Get word to index dictionary
        return text_processing.convert_word_index(all_possible_words_list)

    def create_embeddings(self):

        embeddings = np.zeros((len(word_index_dict), 50))

        ...

    def transform_data():

        ...

    def embeddings_test():

        ...

    def save_transformed_data():

        ...

    def custom_collate():

        ...