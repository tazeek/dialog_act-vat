from torch import nn

class LSTM_GLove(nn.Module):

    def __init__(self, embedding_layer):

        # Number of samples to pass in training
        # Explanation: https://discuss.pytorch.org/t/generic-question-about-batch-sizes/1321
        self._batch_size = 64

        # Output size: Based on labels
        # Layers: Number of layers
        # Hidden Nodes: Number of Nodes
        self._output_size = 4
        self._layers = 1
        self._hidden_nodes = 512

        # Vocab size: Number of unique words
        self._vocab_size = embedding_layer.size(0)
        self._embedding_dim = embedding_layer.size(1)
        
        self._word_embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        self._word_embeddings.weight = nn.Parameter(embedding_layer, requires_grad = False)

        self._lstm = nn.LSTM(self._embedding_dim, self._hidden_nodes)
        self._label = nn.Linear(self._hidden_nodes, self._output_size)