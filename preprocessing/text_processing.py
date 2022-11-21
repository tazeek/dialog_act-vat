from string import punctuation

def _tokenizer(utterance: str):
    return [word.lower() for word in utterance.split(" ") if word != '']

def preprocess_text(utterances: list, remove_punctuation:'bool'=True) -> list:

    processed_utterances = []

    """
        TODO:
        - Complete the sentence (example: i'll -> i, will)
        - Break sentence further (example: jane's -> jane, ', s)
    """

    for utterance in utterances:
        # Split based on tokenizer
        words = _tokenizer(utterance)

        # Remove punctuation (OPTIONAL)
        if remove_punctuation:
            words = [word for word in words if word not in punctuation]

        processed_utterances += [words]


    return processed_utterances

def convert_word_index(words_list: list) -> dict:

    word_to_index = {0: '<pad>'}
    index = 1
    
    for list in words_list:

        for word in list:

            # Check if word does not exist
            # and Increment index
            if word not in word_to_index:
                word_to_index[word] = index
                index += 1

    print(f"Number of unique words: {len(word_to_index)}")

    return word_to_index

def transform_text_integer(words_list: list, index_dict: dict) -> list:

    transformed_list = []

    for words in words_list:
        transformed_list += [[index_dict[word] for word in words]]
    
    return transformed_list