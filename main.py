from data_loaders import dailydialog_full
from preprocessing import text_processing, glove_embeddings, prepare_datasets

import model
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
    word_to_index = _get_index_dictionary()

    # Create lookup table and check for embeddings test
    glove_embeddings = glove_embeddings.create_glove_embeddings(word_to_index)

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

    # Train the model
    train_model = model.train_model(train_generator, valid_generator, glove_embeddings, base_filename)

    # Test the model
    model.test_model(test_generator, train_model, base_filename)