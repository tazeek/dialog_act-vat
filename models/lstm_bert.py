from torch import nn
from transformers import BertModel

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

        # Create BERT embedding layer
        self._bert = BertModel.from_pretrained('bert-base-uncased')

        # Create LSTM model and linear layer
        self._lstm = nn.LSTM(
            input_size = 758, 
            hidden_size = self._hidden_nodes,
            num_layers = self._layers,
            batch_first = True
        )

        self._linear = nn.Linear(self._hidden_nodes, self._output_size)

    def forward(self, input_ids, attention_masks):

        # Extract the features from the BERT model
        hidden_states_output = None

        with torch.no_grad():
            outputs = self._bert(input_ids = input_ids, attention_mask=attention_masks)
        
            _, _, hidden_states_output, _ = outputs
        
        cls_output = hidden_states_output[0][:, 0, :]
        # Input the second transformation to LSTM
        _, (hidden, cell) = self._lstm(cls_output)

        # Get the output in the softmax
        return self._linear(hidden[-1])