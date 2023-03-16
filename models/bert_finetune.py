from torch import nn
from transformers import BertModel

class BERT(nn.Module):
    def __init__(self, output_size):

        super(BERT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, output_size)
        
    def forward(self,ids,mask,token_type_ids):

        _ , o2 = self.bert_model(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids, 
            return_dict=False
        )
        
        out= self.out(o2)
        
        return out