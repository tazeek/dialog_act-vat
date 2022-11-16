from dailydialog_loader import DailyDialog_Loader
from dataloader_da import DataLoader_DA

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

import torch
import text_processing

def _custom_collate(data: list):

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

def _transform_dataloader(dataloader_dataset):
    return DataLoader(dataloader_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=_custom_collate
    )

def fetch_generators(word_to_index: dict):

    # Load the raw datasets
    x_train, y_train = DailyDialog_Loader('train.zip').fetch_dataframe()
    x_val, y_val = DailyDialog_Loader('validation.zip').fetch_dataframe()
    x_test, y_test = DailyDialog_Loader('test.zip').fetch_dataframe()

    # Preprocessing
    x_train = text_processing.preprocess_text(x_train, remove_punctuation=False)
    x_test = text_processing.preprocess_text(x_test, remove_punctuation=False)
    x_val = text_processing.preprocess_text(x_val, remove_punctuation=False)

    # Transform to integer format for lookup
    x_train = text_processing.transform_text_integer(x_train, word_to_index)
    x_test = text_processing.transform_text_integer(x_test, word_to_index)
    x_val = text_processing.transform_text_integer(x_val, word_to_index)

    # Convert to DataLoaders
    train_set = DataLoader_DA(x_train, y_train)
    train_generator = _transform_dataloader(train_set)

    test_set = DataLoader_DA(x_test, y_test)
    test_generator = _transform_dataloader(test_set)

    valid_set = DataLoader_DA(x_val, y_val)
    valid_generator = _transform_dataloader(valid_set)

    return train_generator, test_generator, valid_generator