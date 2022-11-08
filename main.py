from label_loader import LabelLoader

if __name__ == '__main__':

    # Load the data (DailyDialog)
    label_loader = LabelLoader().fetch_dataframe()

    # Differentiate labels and text
    x_train, y_train = label_loader['dialog_act'], label_loader['utterance']

    # Preprocess the data (DailyDialog)

    # Prepare the model

    # Create the VAT formula

    # Train the model

    # Test the model