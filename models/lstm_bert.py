from torch import nn
from transformers import BertModel

import torch

class Bert_Classifier(nn.Module):

    def __init__(self, args):

        super(Bert_Classifier, self).__init__()

        # Output size: Based on labels
        # Layers: Number of layers
        # Hidden Nodes: Number of Nodes
        self._output_size = args['output_size']
        self._input_siE = args['embedding']

        self._dropout = nn.Dropout(0.5)
        self._linear = nn.Linear(768, self._output_size)
        self._relu = nn.ReLu()

    def forward(self, x_train):

        # Extract the embeddings from model

        # Get the output with ReLu
        ...