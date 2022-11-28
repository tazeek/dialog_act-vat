from torch import nn

import torch

class LSTM_Bert(nn.Module):

    def __init__(self):

        super(LSTM_Bert, self).__init__()

        # Output size: Based on labels
        # Layers: Number of layers
        # Hidden Nodes: Number of Nodes
        self._output_size = 4
        self._layers = 1
        self._hidden_nodes = 256

        # Create BERT tokenizer

        # Create BERT lookup table

        # Create BERT embedding layer

        # Create LSTM model and linear layer

    def forward(self):

        # Transform from raw string to BERT-based

        # Extract the features from the BERT model

        # Enter the features into the LSTM model

        # Extract output into the linear layer

        ...