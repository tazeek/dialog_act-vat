from transformers import BertTokenizer, BertModel

class Bert_Processor:

    def __init__():

        ...

    def _load_model(self):

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained('bert-base-uncased')
        self._model.eval()

    def _tokenize_utterance(self):
        """
            Convert a sentence to a list of integers
            PS: Bert Tokenizer returns other information as well
        """

        ...

    def _perform_transformation(self):
        """
            Convert a list of integers to BERT features
            PS: BERT model returns other pieces of informationa as well
        """
        ...

    def _extract_features(self):
        """
            Extract the specific features required for ML
        """

        ...