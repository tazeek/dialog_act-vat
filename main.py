from label_loader import LabelLoader
from lstm_glove import LSTM_GLove
from dataloader_da import DataLoader_DA

import utils

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

    # Differentiate labels and text
    x_train, y_train = label_loader['utterance'], label_loader['dialog_act']

    # Preprocess the text and get tokens (DailyDialog)
    x_train = utils.preprocess_text(x_train, remove_punctuation=False)

    # Get word to index dictionary
    #word_to_index = utils.convert_word_index(x_train)
    #print(f"Number of unique words: {len(word_to_index)}")

    # Load GloVe model
    #glove_model = utils.load_glove_model()
    #print(f"Length of GloVe model: {len(glove_model)}")

    # Create lookup table and check for embeddings test
    #glove_embeddings = utils.create_glove_embeddings(glove_model, word_to_index)
    #utils.embeddings_test(glove_embeddings, glove_model, word_to_index, 'hungry')

    # Prepare the model
    #model = LSTM_GLove(glove_embeddings)

    # Use Dataset class for PyTorch
    train_set = DataLoader_DA(x_train, y_train)
    utils.transform_dataloader(train_set)

    # Break up the dataset into training and testing

    # Transform the data based on the lookup table

    # Create the VAT formula

    # Train the model

    # Test the model