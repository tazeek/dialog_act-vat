from label_loader import LabelLoader
from dailydialog_loader import DailyDialog_Loader

import utils
import text_processing
import glove_embeddings
import prepare_datasets

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

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
    
    # Train the model
    model = utils.train_model(train_generator, glove_embeddings)

    # Test the model
    #utils.test_model(test_generator, model)

    # Create the VAT formula