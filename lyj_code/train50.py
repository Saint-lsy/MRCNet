import time
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
import torchvision.transforms as transforms
import numpy as np
from sklearn import metrics
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from utils import *  # 自定义dataloader

from nets import resnet


def fix_random(i):
    # seed = 42
    seed = i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    # 但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_loader(root_path, BATCH_SIZE):
    train_data = LNM_trainDataset(r"./train_8_1.csv", root_path, 'over-sample',
                                  transform=None)
    trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    training = LNM_testDataset(r"./train_8_1.csv", root_path, transform=None)
    trainingloader = DataLoader(dataset=training, batch_size=BATCH_SIZE)

    valid_data = LNM_testDataset(r"./test_8_1.csv", root_path, transform=None)
    validloader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    # 6中心的外部验证
    test_data = LNM_testDataset(r'./exter1.csv', root_path, transform=None)
    testloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    # 2个中心的外部验证
    exter_data = LNM_testDataset(r'./exter1.csv', root_path, transform=None)
    exterloader = DataLoader(dataset=exter_data, batch_size=BATCH_SIZE)
    # 4个中心的外部验证
    exter_data2 = LNM_testDataset(r'./exter2.csv', root_path, transform=None)
    exterloader2 = DataLoader(dataset=exter_data2, batch_size=BATCH_SIZE)

    return trainloader,trainingloader,validloader,testloader,exterloader,exterloader2

def pretrained_loader(weight_path):
    # "/home/liuyujia/project/classification-LNM/weight/0831/resnet/resnet831_RMS085_i54_epoch01_tra0.755_val0.705.pth"
    pretrained_dict = torch.load(weight_path)
    model_dict = net.state_dict()

    pretrained_dict = {k[:]: v for k, v in pretrained_dict.items() if k[:] in model_dict}

    if len(pretrained_dict) > 1:
        print('加载预训练权重成功', weight_path)
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

def train(trainloader,optimizer):
    net.train()
    sum_loss = 0.0
    correct = 0
    total = 0.0
    train_probs = torch.Tensor([]).to(device)
    train_labels = torch.IntTensor([]).to(device)

    for i, data in enumerate(trainloader, 0):
        net.train()
        # 准备数据
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # 每训练1个batch打印一次loss和准确率
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).to(device).sum()

        train_probs = torch.cat((train_probs, outputs.data[:, 1]), 0)
        train_labels = torch.cat((train_labels, labels.int()), 0)

    print('[i:%d, epoch:%d, iter:%d] Loss: %.03f  '
          % (line, epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1)))

def validate_auc(validloader,type):
    # 每训练完一个epoch测试一下AUC
    #print("Waiting valid!")
    with torch.no_grad():
        valid_loss = 0
        correct = 0
        total = 0

        valid_probs = torch.Tensor([]).to(device)
        valid_labels = torch.IntTensor([]).to(device)
        for data in validloader:
            net.eval()
            images, labels = data
            images, labels = images.float().to(device), labels.to(device)
            outputs = net(images)
            batch_loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            valid_loss += batch_loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum()
            valid_probs = torch.cat((valid_probs, outputs.data[:, 1]), 0)

            valid_labels = torch.cat((valid_labels, labels.int()), 0)

        print('%s: Loss: %.03f | Acc: %.3f%% | AUC: %.03f '
              % (str(type),valid_loss / len(validloader), (100. * correct / total)
                 , metrics.roc_auc_score(valid_labels.cpu(), valid_probs.cpu())))

    auc_val = metrics.roc_auc_score(valid_labels.cpu(), valid_probs.cpu())
    return auc_val




# 训练
if __name__ == "__main__":
    #for line in range(35):
    count=0
    line = 100
    maxauc = 0
    while(line<400):

        line=line+1
        #print(line+1)
        fix_random(line)

        print("cuda:", torch.cuda.is_available())
        # torch.backends.cudnn.enabled = False
        # 定义是否使用GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        device = torch.device('cuda:0')
        outfolder = "/data/lyj/LNM202203/weight"

        # 超参数设置
        EPOCH = 200
        pre_epoch = 0
        BATCH_SIZE = 64
        LR1 =5e-4
        weight_decay1 = 5e-2
        LR2 = 1e-4
        weight_decay2 = 5e-1
        k = 0
        # 数据加载
        root_path = "/data/lyj/LNM202203/image"+"/"
        trainloader,trainingloader,validloader,testloader,exterloader,exterloader2 = data_loader(root_path,BATCH_SIZE)


        # 模型定义-ResNet
        net = resnet.resnet50().to(device)
        #net = senet.seresnet18().to(device)
        #net = resnext.resnext18().to(device)
        #print(net)
        #net = encoding.models.get_model('ResNeSt50', pretrained=True)
        #net = nn.DataParallel(net, device_ids=device_ids)
        #net = models.vgg19(pretrained=True).to(device)
        #print(net)

        # 读取参数
        pretrained = False
        if pretrained == True:
            weight_path = "/data/lyj/LNM_tumor/weight/nrrd-100_8-1/Resnet18-new/se100-i9_E05_tra0.794_val0.747_test0.725_e0.677_ex0.677.pth"
            #"/data/lyj/LNM_tumor/weight/nrrd-100_8-1/senet/se100-i1_E04_tra0.646_val0.737_test0.527_e0.557_ex0.513.pth"
            #"/data/lyj/LNM_tumor/weight/nrrd-100_8-1/senet/seROI-i4_E13_tra0.714_val0.708_test0.542_e0.599_ex0.488.pth"
            #"/data/lyj/LNM_tumor/weight/nrrd-100_8-1/resnet50/res50-100-i55_E04_tra0.712_val0.737_e0.564_ex0.503.pth"
            #"/data/lyj/LMN07/weight/resnext/resnet50_i1_epoch17_tra0.780_val0.687.pth"
            #"/data/lyj/LNM_tumor/weight/nrrd-100_8-1/resnet50/i1_E24_tra0.752_val0.702_test0.515_e0.551_ex0.490.pth"
            #"/data/lyj/LMN07/weight/seNets/prse_i48_epoch19_tra0.700_val0.707.pth"
            #"/data/lyj/LNM_tumor/weight/roi-64/se18_i98_E29_tra0.698_val0.679_ext0.651.pth"

            #"/data/lyj/LNM_tumor/weight/nrrd-100_8-1/senet/se100-i17_E09_tra0.599_val0.664_test0.457_e0.497_ex0.421.pth"
            #"/data/lyj/LNM_tumor/weight/nrrd-100_8-1/senet/seROI-i15_E05_tra0.684_val0.694_test0.504_e0.552_ex0.454.pth"

            print(pretrained,weight_path)
            pretrained_loader(weight_path)

        # 定义损失函数和优化方式
        criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
        #criterion = FocalLoss()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=LR1, weight_decay=weight_decay1)
        #optimizer = optim.SGD(net.parameters(), lr=LR,  weight_decay=0.0001)
        optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR2, weight_decay=weight_decay2)


        for epoch in range(pre_epoch, EPOCH):
            print('\nEpoch: %d' % (epoch + 1))
            # 开始训练！！！
            if epoch < 20:
                optimizer = optimizer
            else:
                optimizer = optimizer2
                print("optimizer2")
            train(trainloader,optimizer)

            # 每训练完一个epoch训练集AUC
            #print("get the AUC of training!")
            auc_train = validate_auc(trainingloader,"training")

            # 每训练完一个epoch测试一下AUC
            #print("Waiting internal valid!")
            auc_val = validate_auc(validloader,"internal")

            # print("Waiting exter valid!")
            auc_exter1 = validate_auc(exterloader, "2center")
            # print("Waiting exter valid!")
            auc_exter2 = validate_auc(exterloader2, "4center")
            print("max auc:",maxauc)
            if auc_val> maxauc:
                maxauc = auc_val

            if (auc_val>=0.7):
                #if (auc_train - auc_test<0.1):
                    #if epoch>=2:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/50-i%01d_E%02d_1l%.04f_1w%.03f_2l%.04f_2w%.03f_t%.03f_v%.03f_e%.03f_e%.03f.pth'
                                   % (outfolder, line, epoch + 1, LR1*1000, weight_decay1*100, LR2*1000, weight_decay2*100,
                                      auc_train, auc_val, auc_exter1, auc_exter2))

                        count = count + 1
            print("satisfied number:",count)
            if auc_train > 0.9:
                break

        print("Training Finished, TotalEPOCH=%d" % EPOCH)