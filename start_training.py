from train_model import Model

import util_helper

if __name__ == '__main__':

    # Get the logger
    logger = util_helper.get_logger('model_training')

    # Get parser for command line inputs
    logger.info('Fetching Parser')
    args = util_helper.create_parser()

    # Load the config file settings
    logger.info('Load config file')
    config_settings = util_helper.load_config_file()

    # Get filename
    base_filename = util_helper.get_base_filename(args)

    # Load the transformed datasets
    logger.info('Loading datasets')
    train_set = util_helper.load_transformed_datasets(args, 'trainloader')
    test_set = util_helper.load_transformed_datasets(args, 'testloader')

    if args.vat:
        unlabeled_set = util_helper.load_transformed_datasets(args, 'unlabeledloader')

    logger.info('Dataset loading completed')

    # Load the model
    logger.info(f'Loading model: {args.model}')
    model_obj = util_helper.load_model(config_settings, args)

    # Populate all in a dictionary for training model object
    args_dict = {
        'train': train_set,
        'test': test_set,
        'unlabeled': unlabeled_set,
        'file_name': base_filename,
        'model': model_obj,
        'embed': args.embed
    }

    # Create model object
    train_model = Model(args_dict | config_settings, logger)

    # [OPTIONAL] Train the VAT model

    # Train the model [Train set]

    # Test the model [Test set]