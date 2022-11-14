from label_loader import LabelLoader
from dataloader_da import DataLoader_DA

import utils

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

    # Differentiate labels and text
    full_file, full_labels = label_loader['utterance'], label_loader['dialog_act']

    # Preprocess the text and get tokens for dictionary creation
    all_possible_words_list = utils.preprocess_text(full_file, remove_punctuation=False)

    # Get word to index dictionary
    word_to_index = utils.convert_word_index(all_possible_words_list)
    print(f"Number of unique words: {len(word_to_index)}")

    # Load GloVe model
    glove_model = utils.load_glove_model()
    print(f"Length of GloVe model: {len(glove_model)}")

    # Create lookup table and check for embeddings test
    glove_embeddings = utils.create_glove_embeddings(glove_model, word_to_index)
    #utils.embeddings_test(glove_embeddings, glove_model, word_to_index, 'hungry')

    # Break up the dataset into training and testing
    # TODO: use the zip files to load the information
    x_train, y_train, x_test, y_test = utils.split_training_testing(full_file, full_labels)

    # Preprocessing steps
    x_train = utils.preprocess_text(x_train, remove_punctuation=False)
    x_test = utils.preprocess_text(x_test, remove_punctuation=False)

    # Transform to integer format for lookup
    x_train = utils.transform_text_integer(x_train, word_to_index)
    x_test = utils.transform_text_integer(x_test, word_to_index)

    # Use Dataset class for PyTorch
    # and use dataloader to load
    train_set = DataLoader_DA(x_train, y_train)
    train_generator = utils.transform_dataloader(train_set)

    test_set = DataLoader_DA(x_test, y_test)
    test_generator = utils.transform_dataloader(test_set)
    
    # Train the model
    utils.train_model(train_generator, glove_embeddings)

    # Test the model

    # Create the VAT formula