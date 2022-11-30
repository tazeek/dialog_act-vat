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

    # Fetch the datasets (from raw to data generator format)
    logger.info('Loading dataset generators')
    values = prepare_datasets.fetch_generators(args)
    logger.info('Dataset generators loaded successfully')

    # TODO:
    # - Visualize perturbed vs original results
    # - Compare with existing results and SOTA

    params_dict = {
        'training': values['train_generator'],
        'test': values['test_generator'],
        'valid': values['valid_generator'],
        'file_name': base_filename,
        'args': args
    }

    # Train the model
    logger.info('Training Model')
    model = Model(params_dict)
    #model.start_train_bert()
    #model.start_train_glove()
    
    logger.info('Training done successfully')

    # Test the model
    logger.info('Testing Model')
    #model.test_model_bert()
    #model.test_model_glove()
    logger.info('Testing done successfully')