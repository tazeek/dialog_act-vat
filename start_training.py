import util_helper

if __name__ == '__main__':

    # Get the logger
    logger = util_helper.get_logger('model_training')
    unlabeled_set = None
    
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