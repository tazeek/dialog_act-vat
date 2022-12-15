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
        # Input Size: This is based on the embedding size
        self._output_size = args['output_size']
        self._layers = args['layers']
        self._hidden_nodes = args['hidden_nodes']
        self._input_size = args['embedding']

        # Create the LSTM model
        self._lstm = nn.LSTM(
            input_size = self._input_size, 
            hidden_size = self._hidden_nodes,
            num_layers = self._layers,
            batch_first = True
        )

        self._linear = nn.Linear(self._hidden_nodes, self._output_size)

    def forward(self, x_train, x_origin_len, perturbation=torch.FloatTensor()):

        # Perturbation check (only in VAT checks)
        #if perturbation.nelement() != 0:

        #    x_embed = torch.add(x_embed, perturbation)
        
        # Input the embeddings to the pack padded sequence
        pack_output = pack_padded_sequence(
            x_train, 
            x_origin_len, 
            batch_first=True, 
            enforce_sorted=False
        )

        # Input the second transformation to LSTM
        _, (hidden, cell) = self._lstm(pack_output)
        
        # Get the output in the softmax
        return self._linear(hidden[-1])