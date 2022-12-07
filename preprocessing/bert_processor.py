from transformers import BertTokenizer, BertModel

class Bert_Processor:

    def __init__():

        ...

    def _load_model(self):

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained('bert-base-uncased')
        self._model.eval()

    def _tokenize_utterance(self):

        ...

    def _perform_transformation(self):

        ...

    def _extract_features(self):

        ...