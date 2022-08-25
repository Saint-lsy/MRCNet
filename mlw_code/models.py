import torch
from torch import nn
from torchvision import models
from torchsummaryX import summary

def create_net(model='resnet18', num_classes=2, input_channel=3, inplanes=64):
    '''
    从 torchvision.models 修改, 生成 net
    '''
    net = getattr(models, model)(num_classes=num_classes)  # type: nn.Module
    if input_channel==1:
        if 'resnet' in model:
            net.conv1 = nn.Conv2d(1, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    
    return net