

def _load_glove_model():

    path = 'pre_trained\glove.6B.50d.txt'
    glove = {}

    with open(path, encoding="utf8") as f:

        for line in f.readlines():
            values = line.split()
            
            word = values[0]
            vector = np.array(values[1:], dtype='float32')

            glove[word] = vector
        
    return glove

def create_glove_embeddings(word_index_dict: dict) -> list:

    # Load glove model
    glove_vectors = _load_glove_model()

    embeddings = np.zeros((len(word_index_dict), 50))
    count = 0
    # Loop through word_index_dict
    for word, index in word_index_dict.items():

        # Check if item is in glove_vectors
        if word in glove_vectors:
            count += 1
            embeddings[index] = glove_vectors[word]

    print(f"Number of words found: {count}")

    return torch.from_numpy(embeddings).float()