from train_model import Model

import util_helper

if __name__ == '__main__':

    # Get the logger
    logger = util_helper.get_logger()

    # Get parser for command line inputs
    logger.info('Fetching Parser')
    args = util_helper.create_parser()

    # Get filename
    base_filename = util_helper.get_base_filename(args)

    # Load the transformed datasets

    # Load the model

    # Train the model

    # Test the model