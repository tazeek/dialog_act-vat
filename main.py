from data_loaders import dailydialog_full
from preprocessing import text_processing, glove_embeddings, prepare_datasets
from train_model import Model

import util_helper

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

if __name__ == '__main__':

    # Get the logger
    logger = util_helper.get_logger()

    # Get parser for command line inputs
    logger.info('Fetching Parser')
    args = util_helper.create_parser()

    # Get filename
    base_filename = util_helper.get_base_filename(args)

    # Get word to index dictionary
    if args.embed == 'glove':
        
        logger.info('Loading Word-to-index dictionary')
        word_to_index = _get_index_dictionary()
        logger.info('Word-to-index dictionary loaded successfully')

        # Create lookup table and check for embeddings test
        logger.info('Creating embeddings')
        embeddings = _create_embeddings(word_to_index)
        logger.info('Embeddings created successfully')

    # Fetch the datasets (from raw to data generator format)
    logger.info('Loading dataset generators')
    train_generator, test_generator, valid_generator = prepare_datasets.fetch_generators(word_to_index)
    logger.info('Dataset generators loaded successfully')

    # TODO:
    # - Visualize perturbed vs original results
    # - Compare with existing results and SOTA

    # Train the model
    params_dict = {
        'training': train_generator,
        'test': test_generator,
        'valid': valid_generator,
        'file_name': base_filename,
        'embeddings': embeddings,
        'args': args
    }

    logger.info('Training Model')
    model = Model(params_dict)
    model.start_train()
    
    logger.info('Training done successfully')

    # Test the model
    logger.info('Testing Model')
    model.test_model()
    logger.info('Testing done successfully')