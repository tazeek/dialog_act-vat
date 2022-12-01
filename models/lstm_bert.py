from torch import nn
from transformers import BertModel

import torch

class Bert_Classifier(nn.Module):

    def __init__(self):

        super(Bert_Classifier, self).__init__()

        # Output size: Based on labels
        # Layers: Number of layers
        # Hidden Nodes: Number of Nodes
        self._output_size = 4
        self._bert_size = 768

        # Create BERT embedding layer and set to eval mode
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._bert.eval()

        self._dropout = nn.Dropout(0.5)
        self._linear = nn.Linear(768, self._output_size)
        self._relu = nn.ReLu()

    def forward(self, input_ids, attention_masks):

        # Extract the CLS token from the BERT model
        output  = None

        with torch.no_grad():
            _, output  = self._bert(
                input_ids = input_ids, 
                attention_mask=attention_masks, 
                return_dict=False
            )

        # Dropout and pass to the linear layer
        dropout_output = self._dropout(output)
        linear_output = self._linear(dropout_output)

        # Get the output with ReLu
        return self._relu(linear_output)