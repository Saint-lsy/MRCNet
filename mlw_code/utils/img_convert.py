import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from skimage import transform, exposure, measure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def imgSlice2png(img, indices, pth=r'./', window=[-1024, 1023], norm=False, shift=True):
    """
    将有标注的 slice 切片保存为图片.  
    norm == True 时, 将图片归一化到 [0, 255], 可以人眼观看.
    norm == False 时, shift 才有作用. 此时原始灰度值 shift 后保存, 方便数据处理.

    Parameters
    ----------
    img : 3D image volume. SimpleITK.Image or numpy.ndarray

    indices : 指定需要保存图片的索引. int or list.

    pth : 保存图片的路径.

    window : 加窗的区间. 若为 None 则不加窗.

    norm  : If True, 将图片归一化到 [0, 255], 可以人眼观看. 
            If Fasle, 不进行任何归一化或缩放, 保持数据便于不失真地处理.

    shift : If True, 将图片灰度区间 shift 到 `[0, window[1]-window[0]]`, 可以保存为 I;16B 格式灰度图, 相对节省空间. 
            If False, 不进行平移, 可能灰度值有负值, 此时必须保存为 I 格式灰度图.
    """
    if isinstance(img, sitk.Image):
        if window:
            if shift:
                img = sitk.IntensityWindowing(img, window[0], window[1], 0, window[1]-window[0])  # 截取 [-1024, 1024]
            else:
                img = sitk.IntensityWindowing(img, window[0], window[1], window[0], window[1])  # 截取 [-1024, 1024]
        img = sitk.GetArrayFromImage(img)
    elif isinstance(img, np.ndarray):
        if window:
            if shift:
                img[np.where(img < window[0])] = window[0]
                img[np.where(img > window[1])] = window[1]
                img -= window[0]
            else:
                img[np.where(img < window[0])] = window[0]
                img[np.where(img > window[1])] = window[1]
    img = (img - img.min()) / (img.max() - img.min()) * 255 if norm else img

    if not hasattr(indices, '__iter__'):
        indices = [indices]

    img_slices = []
    for i in indices:
        img_slices.append(img[i])
        if not norm:
            img_i = Image.fromarray(img[i]).convert('I;16B') if shift else Image.fromarray(img[i]).convert('I')  # type: Image.Image
            img_i.save(pth + '_img_%03d.tiff' % i)
        else:
            img_i = Image.fromarray(img[i]).convert('L')      # type: Image.Image
            img_i.save(pth + '_img_%03d.png' % i)
    return img_slices
        

def mask2png(mask, pth=r'./'):
    """
    将有标注 mask 层保存为 0/255 二值图片.
    """
    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)
    indices = np.where(np.sum(mask, axis=(1, 2)) != 0)[0]  # 获得有标注的 slice 索引
    mask_slices = []
    for i in indices:
        mask_slices.append(mask[i]*255)
        mask_i = Image.fromarray(mask[i]*255).convert('L')
        mask_i.save(pth + '_mask_%03d.png' % i)
    return mask_slices, indices


def write_slice_with_mask(img, mask, indices, pth=r'./', view=0, window=[-1024, 1023], equalize_hist=False, cmap=None, alpha=.25):
    """
    输入 sitk.Image 或 numpy.array 格式的 img 和 mask, 保存指定 indices 的图片, 其中标注位置用半透明颜色突出.

    Parameters:
    ----------
    img : 3D image volume. SimpleITK.Image or numpy.ndarray

    mask : 3D mask volume. SimpleITK.Image or numpy.ndarray

    indices : 指定需要保存图片的索引. int or list.

    pth : 保存图片的路径.

    view : 视角.

    window : 加窗的区间. 若为 None 则不加窗.

    equalize_hist : 是否使用直方图均衡

    cmap : 用于显示mask的 color map

    alpha : mask 的透明度
    """
    if not cmap:
        cmap = define_reds_colormap()
    plt.register_cmap(cmap=cmap)

    if isinstance(img, sitk.Image):
        if window:
            img = sitk.IntensityWindowing(img, window[0], window[1], window[0], window[1])  # type: sitk.Image
        spacing = img.GetSpacing()             # x, y, z
        img = sitk.GetArrayFromImage(img)      # type: np.ndarray  # z, x, y
    if equalize_hist:
        img = exposure.equalize_hist(img)  # 直方图均衡化
    img = (img-img.min())/(img.max()-img.min()) * 255

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)    # type: np.ndarray

    if not hasattr(indices, '__iter__'):
        indices = [indices]

    for slice_index in indices:
        # xy 平面, 可用于检查 mask 是否正确
        if view == 0:
            plt.figure(figsize=(img.shape[1]/100, img.shape[2]/100))
            plt.imshow(img[slice_index, :, :], cmap='gray')
            if not mask is False:
                plt.imshow(mask[slice_index, :, :], alpha=alpha, cmap=cmap.name)

        # xz, yz 平面, 可用于观察图像跳层
        elif (view == 1) | (view == 2):
            scale = spacing[2] / spacing[view-1]  # 利用 spacing 信息, 将图片缩放至合适比例
            img_layer = img[::-1, slice_index, :] if view == 1 else img[::-1, :, slice_index]
            img_layer = transform.resize(img_layer, (int(img_layer.shape[0]*scale), img_layer.shape[1]))
            plt.figure(figsize=(img_layer.shape[1]/100, img_layer.shape[0]/100))
            plt.imshow(img_layer, cmap='gray', vmin=0, vmax=255)

            if not mask is False:
                mask_layer = mask[::-1, slice_index, :] if view == 1 else mask[::-1, :, slice_index]
                mask_layer = transform.resize(mask_layer, (int(mask_layer.shape[0]*scale), mask_layer.shape[1]))
                plt.imshow(mask_layer, alpha=alpha, cmap=cmap.name)
        else:
            print("Error: parameter 'view' must be 0, 1, or 2. ")
            return 0

        output_path = pth + '_view%d_%03d.png' % (view, slice_index)

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 去除图片白边
        plt.savefig(output_path)
        # print(output_path, 'ok')
        plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变
        plt.clf()  # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot
        plt.close()  # 关闭 window，如果没有指定，则指当前 window


def define_reds_colormap():
    """
    定义红色 colormap
    """
    # get colormap
    nalphas = 256
    color_array = plt.get_cmap('Reds')(range(nalphas))
    # change alpha values
    color_array[:, -1] = np.linspace(0, 1, nalphas)
    color_array[-1, :] = [1, 0, 0, 1]
    # create a colormap object
    Reds_alpha_objects = LinearSegmentedColormap.from_list(name='Reds_alpha', colors=color_array)
    return Reds_alpha_objects


def get_roi(img_slice, mask_slice, pth=r'./', box_hw=[64, 64], filter_area=30):
    """
    提取并保存 2D mask 切片上的所有连通域对应的 ROI.
    将根据传入 img 的灰度值, 决定保存方式为 L, 还是 I;16B, 还是 I (即 I;32B).

    Parameters
    ----------
    img_slice : 单层图片, numpy.ndarray

    mask_slice : 单层图片, numpy.ndarray

    pth : 保存路径

    min_area : 允许的最小面积, 小于此面积的将被忽略
    
    box_hw : int or [h, w], 设置固定的裁剪边长. 若为 None , 则裁剪外接矩形.
    """
    _max = img_slice.max()
    _min = img_slice.min()
    mask_labeled = measure.label(mask_slice, connectivity=2)  # 8连通区域标记
    # print('regions number:', mask_labeled.max())
    
    for roi_attr in measure.regionprops(mask_labeled):
        area = roi_attr.area

        # 过滤较小的勾画区域
        if area < filter_area:
            continue

        bbox = roi_attr.bbox
        if not box_hw:
            roi_array = img_slice[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        else:
            center = [round((bbox[0]+bbox[2]) / 2), round((bbox[1]+bbox[3]) / 2)]
            roi_array = img_slice[center[0]-32:center[0]+32, center[1]-32:center[1]+32]
        roi_img = Image.fromarray(roi_array)
        if _max > 256:
            roi_img = roi_img.convert('I') if _min < 0 else roi_img.convert('I;16B')
            output_path = pth+'_roi_%d-%d.tiff' % (roi_attr.label, area)
        else:
            roi_img = roi_img.convert('L')
            output_path = pth+'_roi_%d-%d.png' % (roi_attr.label, area)
        
        roi_img.save(output_path)
        # print(np.array(roi_img))
    return len(measure.regionprops(mask_labeled))






# dcm_dir = r'J:\data\2001_NFYY-LNM\test_dcm'
# img_reader = sitk.ImageSeriesReader()
# dcm_series = img_reader.GetGDCMSeriesFileNames(dcm_dir)
# img_reader.SetFileNames(dcm_series)
# img = img_reader.Execute()
# img = sitk.GetArrayFromImage(img)


# mask_pth = r'.\test_dcm\9-Venous Phase  1.5  B30f single.nii.gz'
# mask = sitk.ReadImage(mask_pth)
# mask = sitk.GetArrayFromImage(mask)

# mask_slices, indices = mask2png(mask)
img_slices = imgSlice2png(img, indices, window=[-1024, 1024], norm=False, shift=True)
# get_roi(img_slices[1], mask_slices[1])
# write_slice_with_mask(img, mask, indices=indices, view=0, alpha=0.25)


