from label_loader import LabelLoader
from dailydialog_loader import DailyDialog_Loader

import model
import text_processing
import glove_embeddings
import prepare_datasets
import argparse

def create_parser():

    parser = argparse.ArgumentParser(description='Parser for VAT and Dialog Act')
    
    return parser.parse_args()

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

    # Create the VAT formula and test:
    # - DailyDialog's validation set
    # - Unlabeled data
    # - Both

    # TODO:
    # - Store train results in a CSV file
    # - Visualize results from CSV file
    # - Add hyperparameters for getting args
    # - Store confusion matrix for labels
    # - Use RoBERTa and BERT embeddings
    # - Visualize perturbed vs original results
    # - Save different types of models
    # - Compare with existing results and SOTA
    # - Check if VAT loss is actually correct or not (Refer to Paper)

    # Later TODO:
    # - Convert model.py to class file

    # Train the model
    train_model = model.train_model(train_generator, valid_generator, glove_embeddings)

    # Test the model
    model.test_model(test_generator, train_model)