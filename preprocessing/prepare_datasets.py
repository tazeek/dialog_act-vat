from data_loaders import dailydialog
from preprocessing.bert_processor import Bert_Processor
from preprocessing.glove_processor import Glove_Processor

import torch

def _preprocess_methods(processor, logger):

    return {
        'glove': Glove_Processor(logger),
        'bert': Bert_Processor()
    }[processor]

def transform_features_datasets(args, logger) -> None:

    # Load the raw datasets
    logger.info('Loading training and testing datasets')
    x_train, y_train = dailydialog.DailyDialog('train.zip').fetch_dataframe()
    x_test, y_test = dailydialog.DailyDialog('test.zip').fetch_dataframe()

    # Load the preprocessor
    processor_method = args.embed

    logger.info(f'Using processor: {processor_method}')
    processor = _preprocess_methods(processor_method, logger)

    # Transform the data and get the data loader
    batch_size = 64

    logger.info(f'Transforming training dataset')
    train_loader = processor.begin_transformation(x_train, y_train, batch_size)

    logger.info(f'Saving training dataset')
    torch.save(train_loader, f'preprocessed_data/trainloader_{processor_method}.pth')

    logger.info(f'Transforming testing dataset')
    test_loader = processor.begin_transformation(x_test, y_test, batch_size * 2)
    
    logger.info(f'Saving testing dataset')
    torch.save(test_loader, f'preprocessed_data/testloader_{processor_method}.pth')

    return None