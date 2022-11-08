def preprocess_text(utterances: list) -> list:

    # Split on whitespace
    processed_utterances = [[utterance.split(' ')] for utterance in utterances]

    return processed_utterances