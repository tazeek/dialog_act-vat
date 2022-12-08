from torch.utils.data import Dataset

class CustomDataLoader(Dataset):

    def __init__(self, text, labels=None):
        
        self._text = text
        self._labels = labels
    
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):

        data_dict = {
            'text': self._text[idx]
        }

        if self._labels is not None:
            data_dict['label'] = int(self._labels[idx])
        
        return data_dict