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

        self._vocab_size
        self._embedding_dim
        
        self._word_embeddings
        self._word_embeddings.weight

        self._lstm
        self._label