from models import bert_finetune

import logging
import argparse
import torch
import tomli

def load_config_file():

    config_dict = {}

    with open("config.toml", mode="rb") as file:
        config_dict = tomli.load(file)

    return config_dict

def get_logger(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] " + \
        "[%(filename)s] " + \
        "[line:%(lineno)d] " + \
        "[%(levelname)s] %(message)s"
    )

    fh = logging.FileHandler(f'log_output/{log_file}.log', 'w')
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def create_parser():

    parser = argparse.ArgumentParser(description='Parser for VAT and Dialog Act')
    
    parser.add_argument('--model', default='lstm', metavar='model',
        help='Name of the model for training')

    parser.add_argument('--embed', default='glove', metavar='embed',
        help='Embeddings required for the model')
    
    parser.add_argument('--vat', default=None, metavar='vat',
        help='Check if the model requires VAT for Semi-supervision')

    return parser.parse_args()

def get_base_filename(args):

    # File name format: <model>_<embed>_<vat>
    base_filename = args.model

    if args.vat:
        base_filename += '_vat'
    
    return base_filename

def load_transformed_datasets(args, file):

    file_name = f'preprocessed_data/{file}_{args.embed}.pth'
    return torch.load(file_name)

def prepare_model_attributes(model):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    return criterion, optimizer

def train_model(train_set):

    # Prepare device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get the model
    model = bert_finetune.BERT_FineTune(768, 5)

    # Load the loss functions
    criterion, optimizer = prepare_model_attributes(model)

    for batch_data in train_set:

        features = batch_data['features']
        labels = batch_data['labels']

        print(features)
        exit()

        input_ids = features['input_ids'].to(device)
        mask = features['attention_mask'].to(device)

def test_model():

    ...