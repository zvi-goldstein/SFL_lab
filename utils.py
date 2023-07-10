import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import one_hot
import torchvision.transforms as T


class BScanDataset(Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_filenames = []
        self.img_dir = img_dir
        self.risk_dirs = os.listdir(self.img_dir)

        for risk_dir in self.risk_dirs:
            self.img_filenames.extend(os.listdir(os.path.join(self.img_dir, risk_dir)))

        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([T.ConvertImageDtype(torch.float)])

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        label = int(os.path.splitext(img_filename)[0][-1])
        try:
            image = read_image(os.path.join(self.img_dir, self.risk_dirs[label], img_filename))
        except RuntimeError:
            print(img_filename)
        if self.transform:
            image = self.transform(image)
        # CAN SUBSTITUTE NUM_CLASSES WITH LENGTH OF RISK DIR. LIST
        label = one_hot(torch.tensor(label), num_classes=2)
        return image, label
