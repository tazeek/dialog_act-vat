from train_model import Model
from preprocessing import prepare_datasets

import util_helper

if __name__ == '__main__':

    # Get the logger
    logger = util_helper.get_logger()

    # Get parser for command line inputs
    logger.info('Fetching Parser')
    args = util_helper.create_parser()

    # Get filename
    base_filename = util_helper.get_base_filename(args)

    # Get word to index dictionary
    logger.info('Loading Word-to-index dictionary')
    
    logger.info('Word-to-index dictionary loaded successfully')

    # Create lookup table and check for embeddings test
    logger.info('Creating embeddings')
    
    logger.info('Embeddings created successfully')

    # Fetch the datasets (from raw to data generator format)
    logger.info('Loading dataset generators')
    train_generator, test_generator, valid_generator = prepare_datasets.fetch_generators(args)
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