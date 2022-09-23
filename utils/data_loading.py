import logging
from math import degrees
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
from skimage.transform import resize
from PIL import Image
import random
from torchvision import transforms

from torch.utils.data.distributed import DistributedSampler
from typing import Sequence
from torch import Tensor
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', transform = None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        img = img.astype(float)
        mask = mask.astype(float)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


class LNM_Dataset(BasicDataset):
    def __init__(self, 
                csv_filepath: str, 
                images_dir = '/data/lsy/LNM_1123/imgs', 
                masks_dir = '/data/lsy/LNM_1123/masks', 
                scale: float = 1.0, 
                mask_suffix: str = '',
                input_channel = 1,
                transform=None,
                mode = 'train'
                ):
        data_info = pd.read_csv(csv_filepath)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = list(data_info['name'])
        self.labels = list(data_info['Class'])
        self.transform = transform
        self.input_channel = input_channel
        self.mode = mode
        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')         

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            #####归一化到0-1  尝试范围  +-512
            img_ndarray = img_ndarray - 1024 
            img_ndarray[img_ndarray < -512] = -512.
            img_ndarray[img_ndarray > 512] = 512.
            img_ndarray = img_ndarray + 512. 
            img_ndarray = img_ndarray / 1024

            #### 范围-1024~+1024
            # img_ndarray = img_ndarray / img_ndarray.max()
            
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            # img_ndarray = img_ndarray / 255

        return img_ndarray    


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        label = self.labels[idx]
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        img = img.astype(float)
        mask = mask.astype(float)
        ####tensfer to 3 channels
        if self.input_channel == 3:
            img = img.repeat(3, axis=0)
            normw = [0.501, 0.501, 0.501]
            normh = [0.175, 0.175, 0.175]
        else:
            normw = [0.501]
            normh = [0.175]
        # .transpose(1, 2, 0)
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        if self.mode == 'train':
            p = np.random.choice([0, 1])#在0，1二者中随机取一个，
            seed = np.random.randint(2147483647)
            random.seed(seed)
            transform_train_img = transforms.Compose([
                transforms.ToPILImage(),#不转换为PIL会报错
                # transforms.Resize(),   
                transforms.RandomRotation(degrees = 5, expand=False, center=None),       
                transforms.CenterCrop(size=(224,224)),
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(p),    ##效果不明显
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.ToTensor(),   
                # transforms.Normalize([0.679, 0.678, 0.678], [0.105, 0.107, 0.108])
                transforms.Normalize(normw, normh)
                ])
            img = transform_train_img(img)                
            random.seed(seed)

            transform_train_seg = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(size=(224,224)),
                transforms.RandomRotation(degrees = 5, expand=False, center=None),       
                # transforms.RandomHorizontalFlip(p),    ##效果不明显
                transforms.ToTensor(),   
                ])
            mask = transform_train_seg(mask)        
        elif self.mode == 'val':
            transform_val_img = transforms.Compose([
                transforms.ToPILImage(),#不转换为PIL会报错
                transforms.CenterCrop(size=(224,224)),
                transforms.ToTensor(),   
                transforms.Normalize([0.501], [0.175])
                ])
            transform_val_seg = transforms.Compose([
                transforms.ToPILImage(),#不转换为PIL会报错
                transforms.CenterCrop(size=(224,224)),
                transforms.ToTensor(),   
                ])
            img = transform_val_img(img)
            mask = transform_val_seg(mask)
        if self.input_channel == 1:
            mask = mask[0]
        return {
            'image': torch.as_tensor(img).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous(),
            'label': torch.as_tensor(label).long().contiguous(),
        }
 


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=1, shuffle=False, num_workers=0,
    #     pin_memory=True)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for i in range(len(train_data)):
        img = train_data[i]['image']
        #求均值方差
        # for j in range(3):
        #     mean[j] += img[j, :, :].mean()
        #     std[j] += img[j, :, :].std()
        mean += img[:, :].mean()
        std += img[:, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print('mean:', mean)
    print('std:', std)
    return list(mean.numpy()), list(std.numpy())






class WeightedRandomSamplerDDP(DistributedSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        data_set: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, data_set, weights: Sequence[float], num_replicas: int, rank: int, num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        super(WeightedRandomSamplerDDP, self).__init__(data_set, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples




if __name__ == '__main__':

    train_set = LNM_Dataset('train_8_1.csv', input_channel = 1)
    print(getStat(train_set))
    train_set[0]
    val_set = LNM_Dataset('validation_8_1.csv')
    