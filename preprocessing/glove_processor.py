import numpy as np

class Glove_Processor:

    def __init__(self):

        self._path = 'pre_trained\glove.6B.50d.txt'
        self._weights = self._load_glove_model()

    def _load_glove_model(self):

        glove = {}

        with open(self._path, encoding="utf8") as f:

            for line in f.readlines():
                
                values = line.split()
                
                word = values[0]
                vector = np.array(values[1:], dtype='float32')

                glove[word] = vector
            
        return glove

    def create_word_index():

        ...

    def transform_data():

        ...

    def embeddings_test():

        ...

    def save_transformed_data():

        ...

    def custom_collate():

        ...