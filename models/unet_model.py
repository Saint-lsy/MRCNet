""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits







# class classify_comb2(nn.Module):

#     def __init__(self, n_classes, in_channels1 = 1024, in_channels2 = 64, mid_channels = 256, reduction = 16):
#         super().__init__()

#         self.cls_conv1 = nn.Sequential(
#             nn.Conv2d(in_channels1, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.cls_conv2 = nn.Sequential(
#             nn.Conv2d(in_channels2, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool2 = nn.AdaptiveAvgPool2d(1) 

#         self.fc = nn.Sequential(
#             nn.Linear(mid_channels, 64, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(64, n_classes, bias=False),
#         )

#     def forward(self, x1, x2):
#         x1 = self.cls_conv1(x1)
#         x2 = self.cls_conv2(x2)
#         b, c, _, _ = x1.size()
#         y = self.avg_pool1(x1).view(b, c, 1, 1)    
#         x = x2 * y.expand_as(x2)            
#         x = self.avg_pool2(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


class mtihead_Unet(UNet):

    def __init__(self, 
                n_channels,
                n_classes, 
                bilinear=False,  
                filters=[64, 128, 256, 512, 1024],      
                seg_task: bool = True,
                cls_task: bool = False
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.out_sigmoid = False if n_classes > 1 else True
        self.bilinear = bilinear
        self.seg_task = seg_task
        self.cls_task = cls_task

        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 2 if bilinear else 1
        # self.down4 = ASPP(filters[3], filters[4]// factor)
        self.down4 = Down(filters[3], filters[4] // factor)
        self.up1 = Up(filters[4], filters[3] // factor, bilinear)
        self.up2 = Up(filters[3], filters[2] // factor, bilinear)
        self.up3 = Up(filters[2], filters[1] // factor, bilinear)
        self.up4 = Up(filters[1], filters[0], bilinear)
        self.outc = OutConv(filters[0], n_classes)
        # self.CFA_module1 = CrossFusionAttention(filters[3])
        # self.CFA_module2 = CrossFusionAttention(filters[2])
        # self.CFA_module3 = CrossFusionAttention(filters[1])
        # self.CFA_module4 = CrossFusionAttention(filters[0])
        if self.cls_task:
            # self.cls_head = classify_head(n_classes)
            # self.cls_bridge = classify_bridge(n_classes)
            self.cls_comb = classify_comb(n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.cls_task:
            # cls_out = self.cls_head(x)
            # cls_out = self.cls_bridge(x5)
            cls_out = self.cls_comb(x5, x)
            return logits, cls_out
        else:
            return logits, None


class mtihead_ResUnet(UNet):

    def __init__(self, 
                n_channels,
                n_classes, 
                bilinear=False,        
                filters=[64, 128, 256, 512, 1024], 
                seg_task: bool = True,
                cls_task: bool = False
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.out_sigmoid = False if n_classes > 1 else True
        self.bilinear = bilinear
        self.seg_task = seg_task
        self.cls_task = cls_task

        self.input_layer = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=1, padding=0),
        )

        self.down1 = Res_Down(filters[0], filters[1])
        self.down2 = Res_Down(filters[1], filters[2])
        self.down3 = Res_Down(filters[2], filters[3])
        factor = 2 if bilinear else 1
        self.down4 = Res_Down(filters[3], filters[4] // factor)
        if self.seg_task:
            self.up1 = Res_Up(filters[4], filters[3] // factor, bilinear)
            self.up2 = Res_Up(filters[3], filters[2] // factor, bilinear)
            self.up3 = Res_Up(filters[2], filters[1] // factor, bilinear)
            self.up4 = Res_Up(filters[1], filters[0], bilinear)
            self.outc = OutConv(filters[0], n_classes)
        if self.cls_task:
            # self.cls_head = classify_head(n_classes)
            # self.cls_bridge = classify_bridge(n_classes)
            self.cls_comb = classify_comb(n_classes)
    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4) 
        if self.seg_task:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        if self.cls_task:
            if self.seg_task:
                cls_out = self.cls_comb(x5, x)
                return logits, cls_out
            else:
                # cls_out = self.cls_head(x)
                cls_out = self.cls_bridge(x5)
                return None, cls_out
        else:
            return logits, None


class MRCNet(UNet):

    def __init__(self, 
                n_channels,
                n_classes, 
                bilinear=False,        
                filters=[64, 128, 256, 512, 1024], 
                seg_task: bool = True,
                cls_task: bool = False
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.out_sigmoid = False if n_classes > 1 else True
        self.bilinear = bilinear
        self.seg_task = seg_task
        self.cls_task = cls_task

        self.input_layer = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=1, padding=0),
        )

        self.down1 = Res_Down(filters[0], filters[1])
        self.down2 = Res_Down(filters[1], filters[2])
        self.down3 = Res_Down(filters[2], filters[3])
        factor = 2 if bilinear else 1
        self.down4 = Res_Down(filters[3], filters[4] // factor)
        self.up1 = MRC_Up(filters[4], filters[3] // factor, bilinear)
        self.up2 = MRC_Up(filters[3], filters[2] // factor, bilinear)
        self.up3 = MRC_Up(filters[2], filters[1] // factor, bilinear)
        self.up4 = MRC_Up(filters[1], filters[0], bilinear)
        self.outc = OutConv(filters[0], n_classes)
        if self.cls_task:
            # self.cls_head = classify_head(n_classes)
            # self.cls_bridge = classify_bridge(n_classes)
            self.cls_comb = classify_comb(n_classes)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4) 
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.cls_task:
            # cls_out = self.cls_head(x)
            # cls_out = self.cls_bridge(x5)
            cls_out = self.cls_comb(x5, x)
            return logits, cls_out
        else:
            return logits, None


