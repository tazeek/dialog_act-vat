from torch import nn
from transformers import BertModel

class BERT_FineTune(nn.Module):
    def __init__(self, output_size):

        super(BERT_FineTune, self).__init__()

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Freeze the layers of BERT model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.out = nn.Linear(768, output_size)
        
    def forward(self, input):

        ids = input['input_ids']
        token_type_ids = input['token_type_ids']
        attention_mask = input['attention_mask']

        _ , pooled_output  = self.bert_model(
            input_ids = ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict=False
        )
        
        out= self.out(pooled_output )
        
        return out