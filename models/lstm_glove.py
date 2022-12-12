from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import torch

class LSTM_GLove(nn.Module):

    def __init__(self, args):

        super(LSTM_GLove, self).__init__()

        # Batch size: https://discuss.pytorch.org/t/generic-question-about-batch-sizes/1321

        # Output size: Based on labels
        # Layers: Number of layers
        # Hidden Nodes: Number of Nodes
        self._output_size = 4
        self._layers = 1
        self._hidden_nodes = 256

        # Create the LSTM model
        self._lstm = nn.LSTM(
            input_size = self._embedding_dim, 
            hidden_size = self._hidden_nodes,
            num_layers = self._layers,
            batch_first = True
        )

        self._linear = nn.Linear(self._hidden_nodes, self._output_size)

    def forward(self, input, actual_batch_len, perturbation=torch.FloatTensor()):
        
        # Transform from raw to embeddings
        x_embed = self._word_embeddings(input)

        # Perturbation check (only in VAT checks)
        if perturbation.nelement() != 0:

            x_embed = torch.add(x_embed, perturbation)
        
        # Input the embeddings to the pack padded sequence
        pack_output = pack_padded_sequence(
            x_embed, 
            actual_batch_len, 
            batch_first=True, 
            enforce_sorted=False
        )


        # Input the second transformation to LSTM
        _, (hidden, cell) = self._lstm(pack_output)
        
        # Get the output in the softmax
        return self._linear(hidden[-1])