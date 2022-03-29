from torch.utils.data import Dataset
from PIL import Image

class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)