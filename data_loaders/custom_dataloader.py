from torch.utils.data import Dataset

class CustomDataLoader(Dataset):

    def __init__(self, text, labels=[]):
        
        self._text = text
        self._labels = labels
    
    def __len__(self):
        return len(self._text)

    def __getitem__(self, idx):

        data_dict = {
            'text': self._text[idx]
        }

        if len(self._labels) != 0:
            data_dict['label'] = int(self._labels[idx])
        
        return data_dict