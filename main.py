from label_loader import LabelLoader
import utils

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

    # Differentiate labels and text
    x_train, y_train = label_loader['utterance'], label_loader['dialog_act']

    # Preprocess the data (DailyDialog)
    x_train = utils.preprocess_text(x_train, remove_punctuation=False)

    # Get word to index dictionary
    word_to_index = utils.convert_word_index(x_train)

    # Load GloVe model
    glove_model = utils.load_glove_vector()
    print(len(glove_model))

    # Prepare the model

    # Create the VAT formula

    # Train the model

    # Test the model