from label_loader import LabelLoader
import utils

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

    # Differentiate labels and text
    x_train, y_train = label_loader['utterance'], label_loader['dialog_act']

    # Preprocess the data and get tokens (DailyDialog)
    x_train = utils.preprocess_text(x_train, remove_punctuation=False)

    # Get word to index dictionary
    word_to_index = utils.convert_word_index(x_train)
    print(f"Number of unique words: {len(word_to_index)}")

    # Load GloVe model
    glove_model = utils.load_glove_vector()
    print(f"Length of GloVe model: {len(glove_model)}")

    # Create lookup table
    glove_embeddings = utils.create_glove_embeddings(glove_model, word_to_index)
    print(f"Length of Glove Embedding: {len(glove_embeddings)}")
    print(glove_embeddings.size())

    # Prepare the model

    # Create the VAT formula

    # Train the model

    # Test the model