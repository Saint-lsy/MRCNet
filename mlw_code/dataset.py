import glob2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd

def make_data_pths(pths, multi_roi=True):
    '''
    得到data的路径
    '''
    data_pths = []
    names = []
    for pth in glob2.glob(pths):
        if not multi_roi:
            name = '_'.join(pth.split('\\')[-1].split('_')[:2])
            # if len(name)<10:
            # print(name)
            if name in names:
                # print(name)
                continue
            else:
                names.append(name)
        data_pths.append(pth)
    # print(names)
    # pd.DataFrame(names).to_csv('lalalla.csv')
    return np.array(data_pths)

class LNMDataSet(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.CenterCrop(56),
                T.ToTensor(),
                T.Normalize(mean=[.5153, ], std=[.0380, ])
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pth = self.dataset[idx]
        img = np.array(Image.open(pth), dtype='float32')/2048
        img = Image.fromarray(img)
        img = self.transform(img)
        target = int(pth.split('\\')[-2].split('_')[0])
        return img, target, pth

    def _get_mean_std(self):
        mean = 0
        std = 0
        for pth in self.dataset:
            img = np.array(Image.open(pth), dtype='float32')/2048
            mean += img.mean()
            std += img.std()
        print(mean/len(self.dataset), std/len(self.dataset))
