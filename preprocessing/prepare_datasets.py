from data_loaders import custom_dataloader, dailydialog
from bert_processor import Bert_Processor
from glove_processor import Glove_Processor

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertTokenizer
import torch

from preprocessing import text_processing, glove_embeddings, bert_embeddings
from data_loaders import dailydialog_full

def _get_index_dictionary():

    # Load the full data of DailyDialog
    label_loader = dailydialog_full.DailyDialog_Full().fetch_dataframe()

    # Differentiate labels and text
    full_file = label_loader['utterance']

    # Preprocess the text and get tokens for dictionary creation
    all_possible_words_list = text_processing.preprocess_text(full_file, remove_punctuation=False)

    # Get word to index dictionary
    return text_processing.convert_word_index(all_possible_words_list)

def _create_embeddings(word_to_index):

    # For GloVe
    return glove_embeddings.create_glove_embeddings(word_to_index)

def _custom_collate_glove(data: list):

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

def _transform_dataloader(dataloader_dataset, batch_size):
    return DataLoader(dataloader_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=_custom_collate_glove
    )
    
def fetch_generators(args):

    # Load the raw datasets
    x_train, y_train = dailydialog.DailyDialog('train.zip').fetch_dataframe()
    x_val, y_val = dailydialog.DailyDialog('validation.zip').fetch_dataframe()
    x_test, y_test = dailydialog.DailyDialog('test.zip').fetch_dataframe()

    embeddings = None
    train_generator = None
    test_generator = None
    valid_generator = None

    # Transform to integer format for lookup (if using GloVe)
    if args.embed == 'glove':

        word_to_index = _get_index_dictionary()
        embeddings = _create_embeddings(word_to_index)

        x_train = text_processing.preprocess_text(x_train, remove_punctuation=False)
        x_test = text_processing.preprocess_text(x_test, remove_punctuation=False)
        x_val = text_processing.preprocess_text(x_val, remove_punctuation=False)

        x_train = text_processing.transform_text_integer(x_train, word_to_index)
        x_test = text_processing.transform_text_integer(x_test, word_to_index)
        x_val = text_processing.transform_text_integer(x_val, word_to_index)

        # Convert to DataLoaders
        train_set = custom_dataloader.DataLoader_DA(x_train, y_train)
        train_generator = _transform_dataloader(train_set, 64)

        test_set = custom_dataloader.DataLoader_DA(x_test, y_test)
        test_generator = _transform_dataloader(test_set, 64)

        valid_set = custom_dataloader.DataLoader_DA(x_val, y_val)
        valid_generator = _transform_dataloader(valid_set, 128)

    # BERT Encoding
    if args.embed == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        x_train, x_train_mask = bert_embeddings.encode_bert(tokenizer, x_train)
        x_val, x_val_mask = bert_embeddings.encode_bert(tokenizer, x_val)
        x_test, x_test_mask = bert_embeddings.encode_bert(tokenizer, x_test)

        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        y_val = torch.tensor(y_val)

        train_tensordataset = TensorDataset(x_train, x_train_mask, y_train)
        test_tensordataset = TensorDataset(x_test, x_test_mask, y_test)
        valid_tensordataset = TensorDataset(x_val, x_val_mask, y_val)

        train_sampler = RandomSampler(train_tensordataset)
        test_sampler = RandomSampler(test_tensordataset)
        valid_sampler = RandomSampler(valid_tensordataset)

        train_generator = DataLoader(train_tensordataset, sampler=train_sampler, batch_size=64)
        test_generator = DataLoader(test_tensordataset, sampler=test_sampler, batch_size=64)
        valid_generator = DataLoader(valid_tensordataset, sampler=valid_sampler, batch_size=128)

    values = {
        'embeddings': embeddings,
        'train_generator': train_generator,
        'test_generator': test_generator,
        'valid_generator': valid_generator
    }

    return values

def _preprocess_methods():

    return {
        'glove': Glove_Processor(),
        'bert': Bert_Processor()
    }

def preprocess_data(args):

    # Load the raw datasets
    x_train, y_train = dailydialog.DailyDialog('train.zip').fetch_dataframe()
    x_test, y_test = dailydialog.DailyDialog('test.zip').fetch_dataframe()

    # Load the preprocessor

    # Transform the data and get the data loader

    # Save using PyTorch
    
    ...