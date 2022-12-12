from models import *

import logging
import argparse
import torch
import tomli

def _models_list(model_name):

    return {
        'lstm_bert': None,
        'lstm_glove': None
    }[model_name]

def load_config_file():

    config_dict = {}

    with open("config.toml", mode="rb") as file:
        config_dict = tomli.load(file)

    return config_dict

def get_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] " + \
        "[%(filename)s] " + \
        "[line:%(lineno)d] " + \
        "[%(levelname)s] %(message)s"
    )

    fh = logging.FileHandler('log_file.log', 'w')
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
    base_filename = args.model + '_' + args.embed

    if args.vat:
        base_filename += '_vat'
    
    return base_filename

def load_transformed_datasets(args, file):

    file_name = f'preprocessed_data/{file}_{args.processor}.pth'
    return torch.load(file_name)

def load_model(config_settings, args):

    # Load the dictionary of models
    default_args = config_settings['default']
    model_args = config_settings[args.model_name]

    # Initialize model and return
    model = _models_list(args.model_name)

    return model(model_args | default_args)