from string import punctuation

def preprocess_text(utterances: list, remove_punctuation:'bool'=True) -> list:

    processed_utterances = []

    for utterance in utterances:
        # Remove whitespace from beginning and end
        processed = utterance.strip()

        # Split on whitespace
        words = processed.split(' ')
        

        # Remove punctuation
        if remove_punctuation:
            words = [word for word in words if word not in punctuation]

        processed_utterances += [words]


    return processed_utterances