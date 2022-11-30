# TODO:
# - Feature embedding
# - Feature extraction
# - Classification

import torch

def encode_bert(tokenizer, data):

    input_ids = []
    attention_mask = []

    for text in data:

        tokenized_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            pad_to_max_length=True,
            padding_side='right',
            return_attention_mask=True
        )

        input_ids.append(tokenized_text['input_ids'])
        attention_mask.append(tokenized_text['attention_mask'])

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    
    return input_ids, attention_mask