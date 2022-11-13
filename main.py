from label_loader import LabelLoader
from lstm_glove import LSTM_GLove
from dataloader_da import DataLoader_DA

import utils

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

    # Differentiate labels and text
    full_file, full_labels = label_loader['utterance'], label_loader['dialog_act']

    # Preprocess the text and get tokens (DailyDialog)
    full_file = utils.preprocess_text(full_file, remove_punctuation=False)

    # Get word to index dictionary
    word_to_index = utils.convert_word_index(full_file)
    print(f"Number of unique words: {len(word_to_index)}")

    # Transform the text to index format
    full_file = utils.transform_text_integer(full_file, word_to_index)

    # Load GloVe model
    glove_model = utils.load_glove_model()
    print(f"Length of GloVe model: {len(glove_model)}")

    # Create lookup table and check for embeddings test
    glove_embeddings = utils.create_glove_embeddings(glove_model, word_to_index)
    #utils.embeddings_test(glove_embeddings, glove_model, word_to_index, 'hungry')

    # Break up the dataset into training and testing
    # TODO: use the zip files to load the information
    utils.split_training_testing(full_file, full_labels)

    # Use Dataset class for PyTorch
    # and use dataloader to load
    train_set = DataLoader_DA(full_file, full_labels)
    train_generator = utils.transform_dataloader(train_set)

    # Create the VAT formula

    # Prepare the model
    #model = LSTM_GLove(glove_embeddings)

    # Train the model
    utils.train_model(train_generator, glove_embeddings)

    # Test the model