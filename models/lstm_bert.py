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
        self._lstm = nn.LSTM(
            input_size = self._embedding_dim, 
            hidden_size = self._hidden_nodes,
            num_layers = self._layers,
            batch_first = True
        )

        self._linear = nn.Linear(self._hidden_nodes, self._output_size)

    def forward(self):

        # Transform from raw string to BERT-based

        # Extract the features from the BERT model

         # Input the second transformation to LSTM
        _, (hidden, cell) = self._lstm(cls_output)

        # Get the output in the softmax
        return self._linear(hidden[-1])