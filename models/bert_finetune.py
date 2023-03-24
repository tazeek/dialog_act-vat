from torch import nn
from transformers import BertModel

class BERT_FineTune(nn.Module):
    def __init__(self, hidden_dim, output_size):

        super(BERT_FineTune, self).__init__()

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Freeze the layers of BERT model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        #self._sent_linear = nn.Linear(hidden_dim, 5)
        self._dialog_act_linear = nn.Linear(hidden_dim, 64)
        self._relu = nn.Relu()
        self._linear = nn.Linear(64, output_size)
        
    def forward(self, ids, attention_mask):

        bert_output  = self.bert_model(
            input_ids = ids,
            attention_mask = attention_mask,
            return_dict=False
        )
        
        out = self._dialog_act_linear(bert_output.pooler_output)
        out = self._relu(out)
        out = self._linear(out)
        
        return out