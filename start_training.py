from train_model import Model

import util_helper

if __name__ == '__main__':

    # Get the logger
    logger = util_helper.get_logger()

    # Get parser for command line inputs
    logger.info('Fetching Parser')
    args = util_helper.create_parser()

    # Load the config file settings
    config_settings = util_helper.load_config_file()

    # Get filename
    base_filename = util_helper.get_base_filename(args)

    # Load the transformed datasets
    train_set = util_helper.load_transformed_datasets(args, 'trainloader')
    test_set = util_helper.load_transformed_datasets(args, 'testloader')

    # Load the model
    model = util_helper.load_model(config_settings, args)

    # [OPTIONAL] Train the VAT model

    # Train the model [Train set]

    # Test the model [Test set]