import random
import cv2
import numpy as np


# 水平翻转
def horizontalFlip(img,p=0.5):
    prob = random.random()
    #print(prob)
    if prob < p:
        new_img = cv2.flip(img,1) #水平旋转
        #print("shuiping")
        return new_img
    else:
        return img

# 竖直翻转
def verticalFlip(img,p=0.5):
    prob = random.random()
    #print(prob)
    if prob < p:
        new_img = cv2.flip(img, 0)  #竖直旋转
        #print("shuzhi")
        return new_img
    else:
        return img

# 归一化
def standardization(data):
    #data[data>1024] = 1024
    #data[data<-1024] = -1024
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

def crop(img,):
    w,h = img.size
    new_img = img[y1:y2, x1:x2]  # type [numpy.ndarray]
    return new_img
