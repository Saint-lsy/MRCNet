import os
from random import sample
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn import metrics
from model import resnet34
from data_loading import LNM_Dataset
from torch.utils.data import DataLoader, random_split

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                      transform=data_transform["train"])
    # train_num = len(train_dataset)

    # # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)



    transform_train = transforms.Compose([
        # transforms.ToPILImage(),#不转换为PIL会报错
        # transforms.Resize(),  
        transforms.ToTensor(),     
        transforms.CenterCrop(size=(300,300)),
        # transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.4)
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.ToTensor(),
        # transforms.Normalize([0.679, 0.678, 0.678], [0.105, 0.107, 0.108])
        ])


    train_set = LNM_Dataset('train_8_1.csv', input_channel=3)
    val_set = LNM_Dataset('validation_8_1.csv')

    train_num = len(train_set)
    val_num = len(val_set)

    batch_size = 32
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size, shuffle=True,
    #                                            num_workers=nw)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform["val"])
    # val_num = len(validate_dataset)
    # validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=batch_size, shuffle=False,
    #                                               num_workers=nw)

    # print("using {} images for training, {} images for validation.".format(train_num,
    #                                                                        val_num))

    # 3. Create data loaders
    '''
    posi_num = 143
    nega_num = 516    
    '''
    train_weights = []
    for train_sample in train_set:
        if train_sample['label']  == torch.tensor(1):
            train_weights.append(659/143)
        else:
            train_weights.append(659/516)     
    train_weights = torch.FloatTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
###### 143posi 516nega
    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    validate_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)   

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "resnet_classification/resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0005)

    epochs = 20
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data['image'], data['label']
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        valid_probs = torch.Tensor([]).to(device)
        valid_labels = torch.IntTensor([]).to(device)
        TP, TN, FP, FN =0, 0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data['image'], val_data['label']
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                TP += torch.sum((predict_y == val_labels.to(device)) & (val_labels.to(device) == 1))
                TN += torch.sum((predict_y == val_labels.to(device)) & (val_labels.to(device) == 0))
                FP += torch.sum((predict_y != val_labels.to(device)) & (val_labels.to(device) == 0))
                FN += torch.sum((predict_y != val_labels.to(device)) & (val_labels.to(device) == 1))

                valid_probs = torch.cat((valid_probs, outputs.data[:, 1]), 0)
                valid_labels = torch.cat((valid_labels, val_labels.to(device)), 0)

        val_accurate = acc / val_num
        sens = TP.double() / (TP + FN)
        spec = TN.double() / (TN + FP)
        val_auc = metrics.roc_auc_score(valid_labels.cpu(), valid_probs.cpu())
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print('validation \n Acc: {:.4f} sens: {:.4f} spec {:.4f} AUC: {:.3f}'.format(val_accurate, sens, spec, val_auc))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
