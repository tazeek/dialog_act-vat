from data_loaders import custom_dataloader

from torch.utils.data import DataLoader
from transformers import BertTokenizer

import torch

class Bert_Processor:

    """
        Preprocessing Steps (from raw to transformed):
        
        1. Load the tokenizer and model
        2. Create the lookup table from dailydialog vocab
        3. Tokenization of utterances from dataset
        4. Convert tokens to integers
        5. Convert the integers to vector tokens via embedding
        6. Save the transformed vector tokens

        Batching for Deep Learning (from transformed to batch):

        1. Load the extracted features and labels
        2. Transform to data loader

        Final output: L * B * N * V

        L -> Number of Hidden State layers from BERT (1 embedding layer, 12 BERT layers)
        B -> Batch Size (Number of utterances)
        N -> Number of tokens
        V -> Vector size of each token (Default: 768)

    """

    def _load_model(self):

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _find_maximum_width(self, text_list):
        # Maximum width refers to the maximum sentence length in the batch

        # Convert from list of strings to list of tokens (Use encode)
        tokens_list = [self._tokenizer.encode(text) for text in text_list]

        return max([len(token_list) for token_list in tokens_list])

    def _custom_collate_fn(self, data):

        text_list, labels_list = zip(*data)

        # Find the maximum length
        max_len = self._find_maximum_width(text_list)

        # Convert from list of strings to list of integers (Use batch_encode_plus)
        # Get the attention masks
        encoded_text = self._tokenizer.batch_encode_plus(
            text_list,
            truncation=True,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'text': text_list,
            'features': encoded_text,
            'labels': torch.tensor(labels_list)
        }

    def begin_transformation(self, text, labels, batch_size):

        # Load the models
        self._load_model()

        # Use customized dataset
        dataset = custom_dataloader.CustomDataLoader(text, labels)

        return DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._custom_collate_fn
        )