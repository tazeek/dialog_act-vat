from dailydialog_loader import DailyDialog_Loader

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

def transform_dataloader(dataloader_dataset):
    return DataLoader(dataloader_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=_custom_collate
    )

def prepare_datasets():

    # Load the raw datasets
    x_train, y_train = DailyDialog_Loader('train.zip').fetch_dataframe()
    x_val, y_val = DailyDialog_Loader('validation.zip').fetch_dataframe()
    x_test, y_test = DailyDialog_Loader('test.zip').fetch_dataframe()

    # Preprocessing
    x_train = text_processing.preprocess_text(x_train, remove_punctuation=False)
    x_test = text_processing.preprocess_text(x_test, remove_punctuation=False)
    x_val = text_processing.preprocess_text(x_val, remove_punctuation=False)

    return x_train, x_val, x_test