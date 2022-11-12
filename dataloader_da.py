from torch.utils.data import Dataset

class DataLoader_DA(Dataset):
    def __init__(self, text, labels):
        ...
    
    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...