from torch.utils.data import Dataset

class CustomDataLoader(Dataset):

    def __init__(self, text, labels=[]):
        
        self._text = text
        self._labels = labels
    
    def __len__(self):
        return len(self._text)

    def __getitem__(self, idx):

        text = self._text[idx]
        label = None

        if len(self._labels) != 0:
            label = int(self._labels[idx])
        
        return text, label