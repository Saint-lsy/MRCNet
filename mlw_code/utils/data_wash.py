import numpy as np
import os
import glob2
import SimpleITK as sitk
from img_convert import imgSlice2png, mask2png, write_slice_with_mask,define_reds_colormap, get_roi
import pandas as pd

# dpyy_dirs = r'.\dpyy\*\V'
gssy_dirs = r'.\gssy\*'
gyzl_dirs = r'.\gyzl\*'
gzsy_dirs = r'.\gzsy\*'
hbsy_dirs = r'.\hbsy\*\V'
# jmzxyy_dirs = r'.\jmzxyy\*'
nbfy_dirs = r'.\nbfy\*'
nczxyy_dirs = r'.\nczxyy\*'
nfyy_dirs = r'.\nfyyBefore\*\ct\vein'
nsrmyy_dirs = r'.\nsrmyy\*'
qdfy_dirs = r'.\qdfy\*\*'
xqyy_dirs = r'.\xqyy\*'
yczxyy_dirs = r'.\yczxyy\*'
yhdyy_dirs = r'.\yhdyy\*'
zdfe_dirs = r'.\zdfe\*'
zdfy_dirs = r'.\zdfy\*'

data_dirs = [gssy_dirs, gyzl_dirs, gzsy_dirs, hbsy_dirs, nbfy_dirs, nczxyy_dirs,
             nfyy_dirs, nsrmyy_dirs, qdfy_dirs, xqyy_dirs, yczxyy_dirs, yhdyy_dirs, zdfe_dirs, zdfy_dirs]


# count = 0
# for dcm_dir in glob2.glob(data_dirs):
#     print('-'*100)
#     # dcm_dir = r'J:\data\2001_NFYY-LNM\test_dcm'
#     try:
# reader = sitk.ImageSeriesReader()
# seriesIDs = reader.GetGDCMSeriesIDs(dcm_dir)
# dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir, seriesIDs[0])
# reader.SetFileNames(dcm_series)
# image = reader.Execute()
# print(dcm_dir, len(seriesIDs))
#     except:
#         count+=1
#         print('ERROR', dcm_dir, 'ERROR', count)
# print("ERROR NUM: ", count)


# reader = sitk.ImageSeriesReader()
# for i_dirs in data_dirs:
#     for dcm_dir in glob2.glob(i_dirs):
#         seriesIDs=reader.GetGDCMSeriesIDs(dcm_dir)
#         # dcm_series=reader.GetGDCMSeriesFileNames(dcm_dir, seriesIDs[0])
#         print(len(seriesIDs), 'dcm_dir_ok')


output_root = r'.\data_converted'
img_reader = sitk.ImageSeriesReader()
red_cmp = define_reds_colormap()

count = 0
hspts = []
names = []
series_nums = []
mask_nums = []
roi_nums=[]

# for hos_dirs in data_dirs:
#     for dcm_dir in glob2.glob(hos_dirs):
#         count += 1
#         if count != 940:
#             continue
#         # 获取mask路径
#         mask_pth = os.path.join(dcm_dir, os.listdir(dcm_dir)[np.where(['nii' in i for i in os.listdir(dcm_dir)])[0][0]])
#         # 获取医院_姓名
#         pth_split = dcm_dir.split('\\')
#         hspt = pth_split[1]
#         name = pth_split[2]
#         case = hspt+'_'+name

#         output_dir = os.path.join(output_root, hspt, case)
#         # if not os.path.exists(output_dir):
#         #     os.makedirs(output_dir)
#         seriesIDs = img_reader.GetGDCMSeriesIDs(dcm_dir)
#         dcm_series = img_reader.GetGDCMSeriesFileNames(dcm_dir, seriesIDs[0])
#         img_reader.SetFileNames(dcm_series)
#         img = img_reader.Execute()  # type: sitk.Image
#         img_size = img.GetSize()
#         img_dirct = img.GetDirection()
#         img_origin = img.GetOrigin()
#         img = sitk.GetArrayFromImage(img)

#         mask = sitk.ReadImage(mask_pth)
#         mask_size = mask.GetSize()
#         mask_dirct = mask.GetDirection()
#         mask_origin = mask.GetOrigin()
#         mask = sitk.GetArrayFromImage(mask)
        
#         if len(seriesIDs)>1:
#             print('WARNING, 该病例文件夹下有多个序列', img_size, mask_size, img_dirct, mask_dirct, img_origin, mask_origin)

#         if (img_size != mask_size) | (img_dirct!=mask_dirct) | (img_origin!=mask_origin):
#             print('WARNING, 该病例影像与勾画文件不匹配', img_size, mask_size, img_dirct, mask_dirct, img_origin, mask_origin)
        
#         mask_slices, indices = mask2png(mask, pth=output_dir+'\\'+case)
#         if len(indices) > 1:
#             print('WARNING, 该病例有多层勾画')

#         img_slices = imgSlice2png(img, indices, window=[-1024, 1024], norm=False, shift=True, pth=output_dir+'\\'+case)

#         write_slice_with_mask(img, mask, indices=indices, view=0, alpha=0.2, cmap=red_cmp, pth=output_dir+'\\'+case)

#         for img_slice, mask_slice, idx in zip(img_slices, mask_slices, indices):
#             roi_num = get_roi(img_slice, mask_slice, pth=output_dir+'\\'+case+'_%03d' % idx)
#         if (len(indices) == 1) & (roi_num > 1):
#             print('WARNING, 该病例单层勾画, 但有多个ROI')

#         hspts.append(hspt)
#         names.append(name)
#         series_nums.append(len(seriesIDs))
#         mask_nums.append(len(indices))
#         roi_nums.append(roi_num)
        
#         print(count, len(seriesIDs), len(indices), roi_num, dcm_dir, hspt, name)  # 面单 茶壶
#         if count%10==0:
#             df = pd.DataFrame([hspts, names, series_nums, mask_nums, roi_nums]).T
#             df.to_csv('勾画统计.csv')


# 在病例文件夹标注 label, 格式: 标签_医院_姓名
pth = r'E:\Desktop\2001_NFYY-LNM\data\data_converted\*\*\*roi*'
df = pd.read_excel('多中心数据整理版.xlsx')
print(df)
import shutil
for d in glob2.glob(pth):
    # name = d.split('\\')[-1]
    # new_name = str(df[df['拼音'] == name]['淋巴结情况'].values[0]) + '_' + name
    # new_d  = os.path.join(os.path.split(d)[0], new_name)
    # shutil.move(d, new_d)
    print(d)



