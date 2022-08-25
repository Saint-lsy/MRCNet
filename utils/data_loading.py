import logging
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
import Augmentation
import random

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

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

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')







# data loader1
def default_loader1(path):
    IMG_PX_SIZE = 224
    data = sitk.GetArrayFromImage(sitk.ReadImage(path))

    data = resize(data, (IMG_PX_SIZE, IMG_PX_SIZE))
    data = data[np.newaxis, :]
    #print(type(data))

    img2 = np.concatenate((data, data, data), axis=0)

    #print(type(img2),img2.shape)

    img2 = Augmentation.standardization(img2)
    img2 = Augmentation.verticalFlip(img2)
    img2 = Augmentation.horizontalFlip(img2)
    #print(data.shape)
    return img2
#data loader2

def png_loader(path):
    img = Image.open(path)
    return img

class LNM_trainDataset(Dataset):
    def __init__(self, data_path,root_path,type='normal', transform=None,target_transform=None, loader= default_loader1):
        data = pd.read_csv(data_path)
        imgs = []
        label_1=[]
        label_0 = []

        for line in range(data.shape[0]):
            label = data.loc[line, 'Class']
            if label ==1:
                label_1.append(line)
            else:
                label_0.append(line)
        random.shuffle(label_0)
        random.shuffle(label_1)
        #root_path = '/data/lyj/LNM_tumor/data/image_nrrd-100/'
        #------- 正常数据加载 -------- #
        if type=='normal':
            for index in range(len(label_0)):
                i = label_0[index]
                path_part = root_path+data.loc[i,'name']+'.nrrd'
                label_part = data.loc[i,'Class']
                #print(label_part,i)
                imgs.append((path_part,float(label_part)))
            for index in range(len(label_1)):
                i = label_1[index]
                path_part = root_path+data.loc[i,'name']+'.nrrd'
                label_part = data.loc[i,'Class']
                #print(label_part, i)
                imgs.append((path_part,float(label_part)))

        # ------- 过采样训练加载（少数样本重复采集，直到和多数样本数量相同） -------- #
        if type=='over-sample':
            print(type)
            for index in range(len(label_0)):
                i = label_0[index]
                path_part = root_path + data.loc[i, 'name'] + '.nrrd'
                label_part = data.loc[i, 'Class']
                imgs.append((path_part, float(label_part)))

                ind = index%len(label_1)
                #print(ind,len())
                i = label_1[ind]
                path_part = root_path + data.loc[i, 'name'] + '.nrrd'
                label_part = data.loc[i, 'Class']
                imgs.append((path_part, float(label_part)))

        # ------- 欠采样训练加载（不完全采集多数样本，使得多数样本数量相同） -------- #
        if type=='under-sample':
            for index in range(len(label_1)):
                i = label_0[index]
                path_part = root_path + data.loc[i, 'name'] + '.nrrd'
                label_part = data.loc[i, 'Class']
                imgs.append((path_part, float(label_part)))

                ind = index
                i = label_1[ind]
                path_part = root_path + data.loc[i, 'name'] + '.nrrd'
                label_part = data.loc[i, 'Class']
                imgs.append((path_part, float(label_part)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path,label = self.imgs[index]
        img = self.loader(img_path)
        #if label==1:
        if self.transform is not None:
            img = self.transform(img)
            #print(type(img))
        return img,int(label)

    def __len__(self):
        return len(self.imgs)




class LNM_testDataset(Dataset):
    def __init__(self, data_path, root_path, transform=None, target_transform=None, loader= default_loader1):
        data = pd.read_csv(data_path)
        #f = open('/data/lyj/LMN07/result/check.csv','a+')
        #f.write('name,label'+'\n')
        imgs = []
        #root_path = '/data/lyj/LNM_tumor/data/image_nrrd-100'
        for i in range(data.shape[0]):
            path_part = root_path+str(data.loc[i,'name'])+'.nrrd'
            label_part = data.loc[i,'Class']
            imgs.append((path_part,float(label_part)))
            #f.write(path_part+','+str(label_part)+'\n')
        #f.close()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path,label = self.imgs[index]
        img = self.loader(img_path)
        #if label==1:
        if self.transform is not None:
            img = self.transform(img)
            #print(type(img))
        return img,int(label)

    def __len__(self):
        return len(self.imgs)


class LNM_Dataset(BasicDataset):
    def __init__(self, csv_filepath: str, images_dir = '/data/lsy/LNM_1123/imgs', masks_dir = '/data/lsy/LNM_1123/masks', scale: float = 1.0, mask_suffix: str = ''):
        data_info = pd.read_csv(csv_filepath)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = list(data_info['name'])
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
            #####归一化到0-1  尝试范围
            # img_ndarray = img_ndarray / 255
            img_ndarray = img_ndarray / img_ndarray.max()
            
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            # img_ndarray = img_ndarray / 255

        return img_ndarray    


if __name__ == '__main__':
    train_set = LNM_Dataset('train_8_1.csv')
    train_set[0]
    val_set = LNM_Dataset('validation_8_1.csv')
    