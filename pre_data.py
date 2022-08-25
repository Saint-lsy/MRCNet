import numpy as np
import os
import pandas as pd
import cv2
import SimpleITK as sitk
from visual_dicoms import sitk_read_dcmseries, norm_img
import matplotlib.pyplot as plt
from skimage import transform, exposure, measure
from PIL import Image

meta_path = './lsy_workspace'
data_path = 'data/data_converted/'
newimg_path = 'data/data_lsy/imgs'
newseg_path = 'data/data_lsy/masks'
newpngs_path = 'data/data_lsy/pngs'


train_data = pd.read_csv('lsy_workspace/train_8_1.csv')
val_data = pd.read_csv('lsy_workspace/validation_8_1.csv')
exter1_data = pd.read_csv('lsy_workspace/exter1.csv')
exter2_data = pd.read_csv('lsy_workspace/exter2.csv')



# if not os.path.exists('data/data_lsy/jmzxyy'):
#     # os.mkdir创建一个，os.makedirs可以创建路径上多个
#     os.makedirs('data/data_lsy/jmzxyy')



#########所有医院数据进行数据清洗
hosp_name = exter2_data['name'].str.split("_",expand=True)[0]
patient_path = data_path + hosp_name + '/' + exter2_data['name']

for info in patient_path:
    if 'jmzxyy' in info or 'dpyy' in info:
        continue
    if 'zhengxianling' in info:
        ori_path = r'E:\LNM_Pred\data\data_original\nbfy\zhengxianling' 
        seg_path = r'E:\LNM_Pred\data\data_original\nbfy\zhengxianling\zhengxianling.nii.gz' 
        dicom_imgs = sitk_read_dcmseries(ori_path)
        segs = sitk.ReadImage(seg_path)
        segs = sitk.GetArrayFromImage(segs)
        for slice in range(len(segs)):
            if segs[slice].any()==True:
                img = dicom_imgs[slice]
                seg = segs[slice]
                if seg.shape != (512, 512):
                    img = img[-512:,:]
                    seg = seg[-512:,:]
        window=[-1024, 1023]
        img[np.where(img < window[0])] = window[0]
        img[np.where(img > window[1])] = window[1]
        img -= window[0]
        print('image_max is ',img.max(),'min is', img.min())

        hosp_patient_name = info.split('/')[3] + '.npy'
        np.save(os.path.join(newimg_path, hosp_patient_name), img)
        np.save(os.path.join(newseg_path, hosp_patient_name), seg)
        plt.imsave(os.path.join(newpngs_path, info.split('/')[3] +'.png'), img, cmap = 'gray')
        continue
     
    ######筛选出ROI最大的一张seg
    if len(os.listdir(info)) != 4:
        seg_files = [seg_info for seg_info in os.listdir(info) if 'mask' in seg_info]
        one_oreas = []
        for segfile in seg_files:
            segfile_path = info + '/' + segfile
            seg_info = cv2.imread(segfile_path, cv2.IMREAD_GRAYSCALE)
            one_orea = np.count_nonzero(seg_info)
            one_oreas.append(one_orea)
        mask_path = seg_files[one_oreas.index(max(one_oreas))]
        max_num = mask_path.split("_", -1)[3][:3]
        img_names = [img_info for img_info in os.listdir(info) if 'img' in img_info and 'tiff'in img_info]
        assert len(seg_files) == len(img_names), 'nums not match'        
        img_path = [img_info for img_info in os.listdir(info) if 'img' in img_info and 'tiff'in img_info and max_num in img_info][0]

    ######保存所有seg CT用于训练
    # seg_files = [seg_info for seg_info in os.listdir(info) if 'mask' in seg_info]
    # ori_files = [img_info for img_info in os.listdir(info) if 'img' in img_info and 'tiff' in img_info]
    # assert len(seg_files)==len(ori_files),'seg and image is not matched'
    else:
        mask_path = [seg_info for seg_info in os.listdir(info) if 'mask' in seg_info][0]
        img_path = [img_info for img_info in os.listdir(info) if 'img' in img_info and 'tiff' in img_info][0]

    mask_path = os.path.join(info, mask_path)
    img_path = os.path.join(info, img_path)
    # tif = TIFF.open(img_path, mode='r')
    # image = tif.read_image()
    image = Image.open(img_path)
    image = np.array(image, dtype='float32')
    print('image_max is ',image.max(),'min is', image.min())
    #     MIN_BOUND = image.min()
    #     MAX_BOUND = image.max()
    #     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 1
    assert mask.shape == image.shape , 'shape not match'
    hosp_patient_name = info.split('/')[3] + '.npy'

    np.save(os.path.join(newimg_path, hosp_patient_name), image)
    np.save(os.path.join(newseg_path, hosp_patient_name), mask)
    plt.imsave(os.path.join(newpngs_path, info.split('/')[3] +'.png'), image, cmap = 'gray')

    print(info, 'finish')




def norm_img(image): # 归一化像素值到（0，255）之间，且将溢出值取边界值
    MIN_BOUND = image.min()
    MAX_BOUND = image.max()
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) 
    image[image > 255] = 255.
    image[image < 0] = 0.

    return image




######获取江门中心医院  大坪医院   分割 
#####################jmzxyy
base_path = 'data/data_original/jmzxyy'
seg_base_path = 'data/data_original/jmzxyy_seg'
patients_train = train_data[train_data['hospital']=='jmzxyy']
patients_val = val_data[val_data['hospital']=='jmzxyy']
patients_info = pd.concat([patients_train, patients_val], axis=0)

for index, patient_info in patients_info.iterrows():
    ####大坪医院多个'V'文件夹
    ori_path = os.path.join(base_path, patient_info['patient']) 
    seg_path = [os.path.join(seg_base_path, seg_info) for seg_info in os.listdir(seg_base_path) if patient_info['patient'] in seg_info][0]
    dicom_imgs = sitk_read_dcmseries(ori_path)

    segs = sitk.ReadImage(seg_path)
    segs = sitk.GetArrayFromImage(segs)
    if len(segs)!=len(dicom_imgs):
        print(index, 'img and seg not match')
    for slice in range(len(segs)):
        if segs[slice].any()==True:
            img = dicom_imgs[slice]
            seg = segs[slice]
            print(slice, index)
            if seg.shape != (512, 512):
                print(index, 'shape not normal', seg.shape)
#####shift  调整至【0，2048】
    window=[-1024, 1024]
    img[np.where(img < window[0])] = window[0]
    img[np.where(img > window[1])] = window[1]
    img -= window[0]
    print('image_max is ',img.max(),'min is', img.min())
    hosp_patient_name = patient_info['name'] + '.npy'
    np.save(os.path.join(newimg_path, hosp_patient_name), img)
    np.save(os.path.join(newseg_path, hosp_patient_name), seg)
    plt.imsave(os.path.join(newpngs_path, patient_info['name'] +'.png'), img, cmap = 'gray')

################## dpyy
base_path = 'data/data_original/dpyy'
seg_base_path = 'data/data_original/dpyy_seg'
patients_info = exter2_data[exter2_data['hospital']=='dpyy']

for index, patient_info in patients_info.iterrows():
    ####大坪医院多个'V'文件夹
    ori_path = os.path.join(base_path, patient_info['patient'], 'V') 
    seg_path = [os.path.join(seg_base_path, seg_info) for seg_info in os.listdir(seg_base_path) if patient_info['patient'] in seg_info][0]
    dicom_imgs = sitk_read_dcmseries(ori_path)

    segs = sitk.ReadImage(seg_path)
    segs = sitk.GetArrayFromImage(segs)
    if len(segs)!=len(dicom_imgs):
        print(index, 'img and seg not match')
    for slice in range(len(segs)):
        if segs[slice].any()==True:
            img = dicom_imgs[slice]
            seg = segs[slice]
            print(slice, index)
            if seg.shape != (512, 512):
                print(index, 'shape not normal', seg.shape)
                img = img[-512:,:]
                seg = seg[-512:,:]
#####shift  调整至【0，2048】
    window=[-1024, 1024]
    img[np.where(img < window[0])] = window[0]
    img[np.where(img > window[1])] = window[1]
    img -= window[0]
    print('image_max is ',img.max(),'min is', img.min())
    hosp_patient_name = patient_info['name'] + '.npy'
    np.save(os.path.join(newimg_path, hosp_patient_name), img)
    np.save(os.path.join(newseg_path, hosp_patient_name), seg)
    plt.imsave(os.path.join(newpngs_path, patient_info['name'] +'.png'), img, cmap = 'gray')





