from transformers import BertTokenizer, BertModel

class Bert_Processor:

    def __init__():

        ...

    def _custom_collate_fn(self, data):

        text_list = data['text']
        labels_list = data['label']

        # Convert from list of strings to list of tokens (Use encode)
        tokens_list = [self._tokenizer.encode(text) for text in text_list]

        # Find the maximum length
        max_len = max([len(token_list) for token_list in tokens_list])

        # Convert from list of strings to list of integers (Use batch_encode_plus)
        # Get the attention masks
        encoded_text = self._tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_tensors='pt',
            return_attention_masks=True
        )

        # Convert from list of integers to BERT output

        # Extract the given output

        return encoded_text, torch.tensor(labels_list)

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

    def begin_transformation(self, utterances, labels, batch_size):

        # Use custom dataset

        # Use dataloader
        
        ...