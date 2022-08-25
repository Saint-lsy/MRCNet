import os
import warnings

import torch
import numpy as np
from torchvision import transforms as T
import random

class DefaultConfig(object):
    def __init__(self):
        self.env = 'default'  # visdom 环境
        self.vis_port = 8097  # visdom 端口

        self.data_pths = r'E:\Desktop\2001_NFYY-LNM\data\data_converted\*\*\*roi_1*'  # 用于 glob
        self.df_pth = r'E:\Desktop\2001_NFYY-LNM\0216 嘉铭多中心数据整理版.xlsx'
        # self.df_pth = r'E:\Desktop\2001_NFYY-LNM\2002实验用972例.xlsx'
        self.discrete_variable = '淋巴结情况 组织学类型 组织学类型2 组织学分级 FIGO2 HPV情况 CT报宫体侵犯情况 CT报阴道侵犯情况 CT报淋巴结转移情况 '
        self.continuous_variable = '年龄 初潮时间 孕次 产次 身高 体重'

        self.multi_roi = False  # 是否允许单个病例有多个roi(如多层勾画的情况)?
        self.val_ratio = 0.3333  # 训练时 val_ratio=0.2
        self.gpu = 0          # Fasle (使用cpu) or 0 or 1 or 2 ... (gpu_id)  目前只支持cpu or 单GPU
        self.num_workers = 0  # DataLoader 读取数据时使用

        self.model = 'resnet18'
        self.model_pth = None
        self.model_pth = r'checkpoints\1\resnet18_580_7211_0202_18-45-02.pth'

        self.input_channel = 1
        self.batch_size = 16
        self.max_epoch = 10000

        self.optim = 'Adam'         # 'SGD' or 'Adam'
        self.lr = 5e-4
        self.momentum = 0.9         # 当 optim 为 SGD 时有效
        self.weight_decay = 1e-2    # L2正则化
        # self.lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay

        self.transform_train = T.Compose([
            T.RandomRotation(10),
            T.RandomCrop(56),
            # T.CenterCrop(56),
            # T.RandomCrop(48),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[.5153, ], std=[.0380, ])
        ])

        self.transform_val = T.Compose([
            T.CenterCrop(56),
            T.ToTensor(),
            T.Normalize(mean=[.5153, ], std=[.0380, ])
        ])

    def parse(self):
        """
        根据字典kwargs 更新 config参数
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', help='指定使用 gpu(给出gpu_id) 或 cpu(False)', default=False)
        parser.add_argument('--num_workers', help='指定 DataLoader 使用的 num_workers', default=0)

        args = parser.parse_args()
        self.__dict__.update(vars(args))

        print('> user config:')
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                print('-', k, ':', getattr(self, k))

        if self.gpu is False:
            self.device = torch.device('cpu')
            print('> use cpu')
        else:
            torch.cuda.set_device(int(self.gpu))
            self.device = torch.device('cuda')
            print('> use gpu:', self.gpu)

    def fix_random(self):
        # seed = 42
        seed = 9090
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
        #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


cfg = DefaultConfig()
