def preprocess_text(utterances: list) -> list:

    processed_utterances = []

    for utterance in utterances:
        # Remove whitespace from beginning and end
        processed = utterance.strip()

        # Split on whitespace
        processed_utterances += [processed.split(' ')]

    return processed_utterances