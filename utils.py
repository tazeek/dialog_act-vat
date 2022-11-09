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

def convert_word_index(words_list: list) -> dict:

    word_to_index = {}
    index = 0

    
    for list in words_list:

        for word in list:

            # Check if word does not exist
            # and Increment index
            if word not in word_to_index:
                word_to_index[word] = index
                index += 1

    return word_to_index