from torch import nn
from transformers import BertModel

class BERT(nn.Module):
    def __init__(self, output_size):

        super(BERT, self).__init__()

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Freeze the layers of BERT model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.out = nn.Linear(768, output_size)
        
    def forward(self, input_ids):

        _ , o2 = self.bert_model(
            input_ids,
            return_dict=False
        )
        
        out= self.out(o2)
        
        return out