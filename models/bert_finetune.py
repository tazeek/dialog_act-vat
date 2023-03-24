from torch import nn
from transformers import BertModel

class BERT_FineTune(nn.Module):
    def __init__(self, hidden_dim, da_labels, sc_labels):

        super(BERT_FineTune, self).__init__()

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Freeze the layers of BERT model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # For Dialog Act
        self._dialog_act_linear = nn.Linear(hidden_dim, 64)
        self._relu = nn.ReLU()
        self._linear = nn.Linear(64, da_labels)

        # For Sentiment Classification
        self._sentiment_class_linear = nn.Linear(hidden_dim, 64)
        self._relu = nn.RELU()
        self._linear = nn.Linear(64, sc_labels)


        
    def forward(self, ids, attention_mask):

        bert_output = self.bert_model(
            input_ids = ids,
            attention_mask = attention_mask
        )
        
        da_out = self._dialog_act_linear(bert_output.pooler_output)
        da_out = self._relu(da_out)
        da_out = self._linear(da_out)
        
        return da_out