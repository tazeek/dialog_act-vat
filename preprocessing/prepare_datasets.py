from data_loaders import dailydialog
from bert_processor import Bert_Processor
from glove_processor import Glove_Processor

import torch

def _preprocess_methods(processor):

    return {
        'glove': Glove_Processor(),
        'bert': Bert_Processor()
    }[processor]

def preprocess_data(args) -> None:

    # Load the raw datasets
    x_train, y_train = dailydialog.DailyDialog('train.zip').fetch_dataframe()
    x_test, y_test = dailydialog.DailyDialog('test.zip').fetch_dataframe()

    # Load the preprocessor
    processor = _preprocess_methods(args.processor)

    # Transform the data and get the data loader
    batch_size = 64
    train_loader = processor.begin_transformation(x_train, y_train, batch_size)
    test_loader = processor.begin_transformation(x_test, y_test, batch_size)

    # Save using PyTorch
    torch.save(train_loader, 'preprocessed_data/trainloader.pth')
    torch.save(test_loader, 'preprocessed_data/testloader.pth')

    return None