from torch.utils.data import Dataset

class DataLoader_DA(Dataset):
    def __init__(self, text, labels):
        
        self._text = text
        self._labels = labels
    
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        
        label = self._labels[idx]
        text = self.text[idx]

        return {
            'text': text,
            'class': label
        }