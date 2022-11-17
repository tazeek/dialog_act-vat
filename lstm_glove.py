from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import torch

class LSTM_GLove(nn.Module):

    def __init__(self, embedding_layer):

        super(LSTM_GLove, self).__init__()

        # Number of samples to pass in training
        # Explanation: https://discuss.pytorch.org/t/generic-question-about-batch-sizes/1321
        self._batch_size = 64

        # Output size: Based on labels
        # Layers: Number of layers
        # Hidden Nodes: Number of Nodes
        self._output_size = 4
        self._layers = 1
        self._hidden_nodes = 256

        # Vocab size: Number of unique words
        # Embedding dim: Dimension size of each word
        self._vocab_size = embedding_layer.size(0)
        self._embedding_dim = embedding_layer.size(1)
        
        # Create embedding layer and weights
        # We don't have retrain the gradients again
        self._word_embeddings = nn.Embedding(self._vocab_size, self._embedding_dim, padding_idx=0)
        self._word_embeddings.weight = nn.Parameter(embedding_layer, requires_grad = False)

        # Create the LSTM model
        self._lstm = nn.LSTM(
            input_size = self._embedding_dim, 
            hidden_size = self._hidden_nodes,
            num_layers = self._layers,
            batch_first = True
        )

        self._linear = nn.Linear(self._hidden_nodes, self._output_size)

    def forward(self, input, actual_batch_len, perturbation=None):
        
        # Transform from raw to embeddings
        x_embed = self._word_embeddings(input)

        # Perturbation check (only in VAT checks)
        if perturbation:
            x_embed = torch.add(x_embed, perturbation)
        
        # Input the embeddings to the pack padded sequence
        pack_output = pack_padded_sequence(
            x_embed, 
            actual_batch_len, 
            batch_first=True, 
            enforce_sorted=False
        )

        # Input the second transformation to LSTM
        out_lstm, (hidden, cell) = self._lstm(pack_output)
        #dense_outputs = self._linear(hidden[-1])

        # Get the output in the softmax
        #output = self._act(dense_outputs)
        return self._linear(hidden[-1])