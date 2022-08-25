import time

import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from data import data_split
from data.dataset import LNMDataSet, make_data_pths
from models.models import create_net
from utils.visualize import Visualizer, roc_curve
from utils.my_stat import get_sub_df, clinic_stat

def train(net, loader_train, loader_val, optimizer, cfg):
   # 训练
    net.train()
    for epoch in range(cfg.max_epoch):
        pbar = tqdm(loader_train)
        loss_sum = 0
        for idx, (imgs, targets, pths) in enumerate(pbar):
            # print(targets)
            imgs = imgs.to(cfg.device)
            targets = targets.to(cfg.device)

            # 梯度置零
            optimizer.zero_grad()

            # 正向传播, 反向传播, 权重更新
            results = net(imgs)
            loss = criterion(results, targets)
            loss.backward()
            optimizer.step()

            # 显示
            loss_sum += loss.item()
            pbar.set_description('Epoch: [%d/%d], Loss: %.3f, lr: %f' % (epoch+1, cfg.max_epoch, loss_sum / (idx + 1), cfg.lr))

        # validation 和显示, 及保存模型权重
        vis.plot('loss', loss_sum / (idx + 1))
        if (epoch+1) % 5 == 0:
            meter_train = val(net, loader_train, cfg)
            meter_val = val(net, loader_val, cfg)
            vis.plot('auc', meter_val['auc'])
            vis.log('-',  win='log')
            vis.log('Epoch %d, train auc:' % (epoch+1)+str(meter_train), win='log')
            vis.log('Epoch %d, val   auc:' % (epoch+1)+str(meter_val), win='log')
            print('Epoch %d, train auc:' % (epoch+1), meter_train)
            print('Epoch %d, val   auc:' % (epoch+1), meter_val)

            # 保存模型权重
            if (meter_val['auc'] > 0.7) & (abs(meter_train['auc'] - meter_val['auc']) < 0.15):
                prefix = 'checkpoints/' + cfg.model + '_' + str(epoch+1) + '_' + str(int(meter_val['auc']*10000)) + '_'
                name = time.strftime(prefix + '%m%d_%H:%M:%S' + '.pth').replace(':', '-')
                torch.save(net.state_dict(), name)


@torch.no_grad()
def val(net, loader_val, cfg):
    net.eval()
    TP = FP = FN = TN = 0.0
    scores_all = targets_all = np.array([])

    for idx, (imgs, targets, pths) in enumerate(loader_val):
        # 模型前向传播
        imgs = imgs.to(cfg.device)
        scores = nn.functional.softmax(net(imgs), dim=1)

        # 用于计算 auc
        targets_all = np.append(targets_all, np.array(targets.cpu()))
        scores_all = np.append(scores_all, np.array(scores.cpu())[:, 1])

        # 用于计算 acc, spec, sens, , p, r, f1
        _, preds = (tensor.cpu() for tensor in torch.max(scores, 1))
        # TP    predict 1 label 1
        TP += ((scores.cpu()[:, 1] > 0.17638) & (targets == 1)).sum()
        # FP    predict 1 label 0
        FP += ((scores.cpu()[:, 1] > 0.17638) & (targets == 0)).sum()
        # FN    predict 0 label 1
        FN += ((scores.cpu()[:, 1] <= 0.17638) & (targets == 1)).sum()
        # TN    predict 0 label 0
        TN += ((scores.cpu()[:, 1] <= 0.17638) & (targets == 0)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN)  # accuracy
    spec = TN / (TN + FP)     # specificity
    sens = TP / (TP + FN)     # sensitivity
    p = TP / (TP + FP)        # precision  精确率 真阳性比率
    r = TP / (TP + FN)        # recall  召回率
    f1 = 2 * r * p / (r + p)  # F1 socre
    print(sens, spec, p, TN/(TN+FN), acc)

    auc = roc_auc_score(targets_all, scores_all, )
    net.train()
    print({'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN), 'acc': round(float(acc), 4), 'auc': round(auc, 4),
                   'spec': round(float(spec), 4), 'sens': round(float(sens), 4), 'p': round(float(p), 4), 'r': round(float(r), 4), 'f1': round(float(f1), 4)})
    return targets_all, scores_all
    # return {'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN), 'acc': round(float(acc), 4), 'auc': round(auc, 4),
    #         'spec': round(float(spec), 4), 'sens': round(float(sens), 4), 'p': round(float(p), 4), 'r': round(float(r), 4), 'f1': round(float(f1), 4)}


if __name__ == '__main__':
    cfg.parse()
    cfg.fix_random()

    # 创建 Dataset, DataLoader
    data_pths = make_data_pths(cfg.data_pths, multi_roi=cfg.multi_roi)
    pths_train, pths_val = data_split(data_pths, val_ratio=cfg.val_ratio)
    # print(data_pths)
    print(len(data_pths), len(pths_train), len(pths_val))

    # 临床变量分析
    names_train = list(set(['_'.join(i.split('\\')[-1].split('_')[:2]) for i in pths_train]))
    names_val = list(set(['_'.join(i.split('\\')[-1].split('_')[:2]) for i in pths_val]))
    df_train = get_sub_df(cfg.df_pth, names_train, '拼音')
    df_val = get_sub_df(cfg.df_pth, names_val, '拼音')
    print(len(names_train), len(names_val))
    clinic_stat(df_train, df_val, cfg.discrete_variable, cfg.continuous_variable)
    

    data_train = LNMDataSet(pths_train, transform=cfg.transform_train)
    loader_train = DataLoader(data_train, cfg.batch_size, shuffle=True, num_workers=int(cfg.num_workers))

    data_val = LNMDataSet(pths_val, transform=cfg.transform_val)
    loader_val = DataLoader(data_val, cfg.batch_size, shuffle=True, num_workers=int(cfg.num_workers))

    # 创建 network, loss
    net = create_net(model=cfg.model, input_channel=cfg.input_channel).to(cfg.device)
    if cfg.model_pth:
        net.load_state_dict(torch.load(cfg.model_pth))
        print('> load model from %s.' % cfg.model_pth)

    # 定义 loss, optimizer
    criterion = nn.CrossEntropyLoss()

    if cfg.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    vis = Visualizer(cfg.env, port=cfg.vis_port)

    # train(net, loader_train, loader_val, optimizer, cfg)
    values_train = (targets_train, scores_train) = val(net, loader_train, cfg)
    values_test = (targets_val, scores_val) = val(net, loader_val, cfg)
    roc_curve([values_train, values_test])
