from preprocessing import prepare_datasets

import util_helper

if __name__ == '__main__':

    # Get the logger
    logger = util_helper.get_logger('feature_extraction')
    logger.info('Logger created successfully')

    # Get parser for command line inputs
    logger.info('Fetching Parser')
    args = util_helper.create_parser()

    # Fetch the datasets (from raw to data generator format)
    prepare_datasets.transform_features_datasets(args, logger)