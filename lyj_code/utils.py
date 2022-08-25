from torch.utils.data import Dataset, DataLoader
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
from skimage.transform import resize
from PIL import Image
import utils.Augmentation as Augmentation
import random

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
