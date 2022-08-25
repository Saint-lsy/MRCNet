from matplotlib import image
from numpy.core.fromnumeric import shape
import pylab
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
#pydicom read dcm
def dicom_read():
    ds=pydicom.read_file("C:\\Users\\57575\\Desktop\\I1000000")
    print(ds.dir("pat"))
    pix = ds.pixel_array
    ##读取显示图片
    pylab.imshow(pix, cmap=pylab.cm.bone)
    pylab.show()

def sitk_read_dcmseries(img_path):
    
    reader = sitk.ImageSeriesReader() 
    img_names = reader.GetGDCMSeriesFileNames(img_path) 
    reader.SetFileNames(img_names) 
    image = reader.Execute() 
    image_array = sitk.GetArrayFromImage(image)
    # print('dicom size:',image.GetSize()) #512*512*208
    # spacing = np.array(image.GetSpacing()) 
    # print(spacing)
    '''
    print meta
    '''
    # keys = image.GetMetaDataKeys()
    # print(keys)
    # for key in keys:
    #     print(key, image.GetMetaData(key))
    # print(image_array)
    return image_array

def sitk_read_mha(filename):
    itk_img = sitk.ReadImage(filename) 
    img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
    # num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    # origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    # spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    # print('seg size:',img_array.shape) 
    # print(spacing)


    '''
    print meta
    '''
    # keys = itk_img.GetMetaDataKeys()
    # for key in keys:
    #     print (key, itk_img.GetMetaData(key))
    # print(img_array[0])

    return img_array

def norm_img(image): # 归一化像素值到（0，255）之间，且将溢出值取边界值
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = 255* (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) 
    image[image > 255] = 255.
    image[image < 0] = 0.

    return image


def find_segment_slice(seg_array):
    ##find segment slice
    seg_slice = []
    seg_list = []
    for i in range(seg_array.shape[0]):
        if np.any(seg_array[i,:,]):
            seg_slice.append(i)
            seg_list.append(np.count_nonzero(seg_array[i,:,]))
            # print(np.nonzero(seg_array[90,:,:]))
    return seg_slice[np.argmax(seg_list)]


def  visualize(ori_array, seg_array, num_z = 0, show_3D = True):

    if show_3D == True:
        plt.subplot(3,2,1)
        plt.imshow(ori_array[:,210,:])
        plt.subplot(3,2,2) 
        plt.imshow(seg_array[:,210,:])


        plt.subplot(3,2,3)
        plt.imshow(ori_array[:,:,210])
        plt.subplot(3,2,4) 
        plt.imshow(seg_array[:,:,210])


        plt.subplot(3,2,5)
        plt.imshow(ori_array[num_z,:,:])
        plt.subplot(3,2,6) 
        plt.imshow(seg_array[num_z,:,:])
        plt.show()
    else:
        # plt.subplot(1,2,1)
        # plt.imshow(ori_array[num_z,:,:],cmap='gray', vmin = 0, vmax = 200)
        # plt.axis('off')
        # plt.subplot(1,2,2) 
        plt.imshow(ori_array[num_z,:,:],cmap='gray', vmin = 0, vmax = 200)
        # mask = seg_array[num_z,:,:].T
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i][j] ==1:
        #             plt.scatter(i,j,c = 'yellow',s =1)      
        plt.axis('off')
        plt.imsave('slice_V.png',ori_array[num_z,:,:], dpi=800 ,cmap='gray', vmin = 0, vmax = 200)
        plt.show()

# def visualize_roi(ln3_roi, ln3_array, range):
    # for num in range(3):
    #     plt.subplot(1,2,1)
    #     plt.imshow(dicom_array[ln3_z1 +num, ln3_y1:ln3_y2, ln3_x1:ln3_x2])
    #     plt.subplot(1,2,2) 
    #     plt.imshow(dicom_array[ln3_z1 + num, ln3_y1:ln3_y2, ln3_x1:ln3_x2])
    #     mask = ln3_array[ln3_z1 + num, ln3_y1:ln3_y2, ln3_x1:ln3_x2]
    #     for i in range(mask.shape[0]):
    #         for j in range(mask.shape[1]):
    #             if mask[j][i] ==1:
    #                 plt.scatter(i,j,c = 'yellow',s =1)
    #     plt.show()

# 对医疗图像进行重采样，仅仅需要将out_spacing替换成自己想要的输出即可
def resample_image(itk_image, out_spacing=[1.0, 1.0, 2.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
 
    # 根据输出out_spacing设置新的size
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
 
    resample.SetInterpolator(sitk.sitkBSpline)
 
    return resample.Execute(itk_image)
 
# gz_path = 'PANCREAS_0015.nii.gz'
# print('测试文件名为：', gz_path)
 
# # 使用sitk读取对应的数据
# Original_img = sitk.ReadImage(gz_path)
# print('原始图像的Spacing：', Original_img.GetSpacing())
# print('原始图像的Size：', Original_img.GetSize())
 
# # 对数据进行重采样
# Resample_img = resample_image(Original_img)
# print('经过resample之后图像的Spacing是：', Resample_img.GetSpacing())
# print('经过resample之后图像的Size是：', Resample_img.GetSize())
 

if __name__ =='__main__':
    # filename = 'E:/radiomic data/High invasive HCC/Center 1-ok/bochanghua/orimeta/1.3.12.2.1107.5.1.4.73756.30000014031400331445300006755.mha'
    filename = 'E:/radiomic data/High invasive HCC/Center 2-ok/aojinglai/Orimeta/FILE487.mha'
    # filename2 = 'E:/radiomic data/High invasive HCC/Center 1-ok/bochanghua/segmeta/Untitled.mha'
    filename2 = r'G:\gastic_ENE\v2\TJtumor_batch1\Data1\P000426300\V\V-P000423765-LN3.mha'

    # sitk_read_mha(filename)

    dicom_path = r'G:\gastic_ENE\v2\TJtumor_batch1\Data1\P000426300\V'

    #208 slice   512*512
    dicom_array = sitk_read_dcmseries(dicom_path)
    ori_array = sitk_read_mha(filename)
    seg_array = sitk_read_mha(filename2)
    # print(dicom_array[0])
    # print(ori_array[0])
    # print(np.max(ori_array))
    # print(np.min(ori_array))
    ori_array = norm_img(ori_array)
    # print(ori_array)
    slice = find_segment_slice(seg_array)
    # dicom_array = norm_img(dicom_array)
    visualize(dicom_array,seg_array,slice,show_3D=False)
