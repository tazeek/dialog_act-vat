from label_loader import LabelLoader
from utils import preprocess_text

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

    # Differentiate labels and text
    x_train, y_train = label_loader['utterance'], label_loader['dialog_act']

    print(x_train[0])
    # Preprocess the data (DailyDialog)
    x_train = preprocess_text(x_train, remove_punctuation=True)
    print(x_train)

    # Prepare the model

    # Create the VAT formula

    # Train the model

    # Test the model