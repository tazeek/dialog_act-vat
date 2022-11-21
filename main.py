from data_loaders import dailydialog_full
from preprocessing import text_processing, glove_embeddings, prepare_datasets

import model
import logging
import argparse

def _get_logger():

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

def _create_parser():

    parser = argparse.ArgumentParser(description='Parser for VAT and Dialog Act')
    
    parser.add_argument('--model', default='lstm', metavar='model',
        help='Name of the model for training')

    parser.add_argument('--embed', default='glove', metavar='embed',
        help='Embeddings required for the model')
    
    parser.add_argument('--vat', default=None, metavar='vat',
        help='Check if the model requires VAT for Semi-supervision')

    return parser.parse_args()

def _get_base_filename(args):

    # File name format: <model>_<embed>_<vat>
    base_filename = args.model + '_' + args.embed

    if args.vat:
        base_filename += '_vat'
    
    return base_filename

if __name__ == '__main__':

    # Get the logger
    logger = _get_logger()

    # Get parser for command line inputs
    logger.info('Fetching Parser')
    args = _create_parser()

    # Get filename
    base_filename = _get_base_filename(args)

    # Load the full data of DailyDialog
    label_loader = dailydialog_full.DailyDialog_Full().fetch_dataframe()

    # Differentiate labels and text
    full_file, full_labels = label_loader['utterance'], label_loader['dialog_act']

    # Preprocess the text and get tokens for dictionary creation
    all_possible_words_list = text_processing.preprocess_text(full_file, remove_punctuation=False)

    # Get word to index dictionary
    word_to_index = text_processing.convert_word_index(all_possible_words_list)

    # Create lookup table and check for embeddings test
    glove_embeddings = glove_embeddings.create_glove_embeddings(word_to_index)\

    # Fetch the datasets (from raw to data generator format)
    train_generator, test_generator, valid_generator = prepare_datasets.fetch_generators(word_to_index)

    # Create the VAT formula and test:
    # - DailyDialog's validation set
    # - Unlabeled data
    # - Both

    # TODO:
    # - Use RoBERTa and BERT embeddings
    # - Visualize perturbed vs original results
    # - Compare with existing results and SOTA
    # - Check if VAT loss is actually correct or not (Refer to Paper)

    # Later TODO:
    # - Convert model.py to class file
    # - Add in rest of hyperparameters
    # - Add logging parameters
    # - Check if metrics are calculated properly
    # - Store the CSV results in a folder

    # NEXT: Segment codebase (see paper) -> Add logging parameters

    # Train the model
    train_model = model.train_model(train_generator, valid_generator, glove_embeddings, base_filename)

    # Test the model
    model.test_model(test_generator, train_model, base_filename)