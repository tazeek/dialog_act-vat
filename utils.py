class TextPreprocess:

    def __init__(self):

        # Define the methods used
        self._methods = {
            'word2vec': _process_word_vectors()
        }

    def process_text(self, method: str):

        return self._methods[method]

    def _process_word_vectors(self):

        ...