import argparse
import logging
from pyexpat import model
import re
import sys
import tempfile
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss, dice_coeff, multiclass_dice_coeff
from models import MRCNet, mtihead_Unet, resnet34, resnet50, resnet101, ResUnet, ResUnetPlusPlus, ResNet18, mtihead_ResUnet
from torchvision.models import resnet18
import torch.distributed as dist
import numpy as np
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate, undis_evaluate
import random
from utils.data_loading import LNM_Dataset
from utils.lossfunction import GeneralizedDiceLoss, WeightedCrossEntropyLoss, FocalLossV1
seed = 42
torch.manual_seed(seed)  # cpu种子
# torch.cuda.manual_seed(seed) # 当前GPU的种子
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = True   # 默认值
torch.backends.cudnn.benchmark = False  # 默认为False
torch.backends.cudnn.deterministic = True
# dir_img = Path('/data/lsy/carvana/imgs/train/')
# dir_mask = Path('/data/lsy/carvana/masks/train_masks/')
dir_checkpoint = Path('./checkpoints/')
checkpoint_path = Path('./checkpoints/')
def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 4,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              input_channel = 1,
              seg_task: bool = True,
              cls_task: bool = False,
              ):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_set = LNM_Dataset('dataset/train_10_12.csv',input_channel=input_channel)
    val_set = LNM_Dataset('dataset/validation_10_12.csv', input_channel=input_channel, mode='val')

    n_train = len(train_set)
    n_val = len(val_set)


    '''
    posi_num = 159
    nega_num = 620   
    all_num = 779 
    '''
    train_weights = []
    for train_sample in train_set:
        if train_sample['label']  == torch.tensor(1):
            train_weights.append(779/159)
        else:
            train_weights.append(779/620)     
    train_weights = torch.FloatTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
###### 143posi 516nega



    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 3. Create data loaders
    # , sampler=train_sampler
    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=nw, pin_memory=True) 


    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))


    tb_writer = SummaryWriter()
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                amp=amp))
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type if hasattr(device, 'type') else device}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 6.]).to(device))
    weightedbceloss = WeightedCrossEntropyLoss(ignore_index=-100)
    GenDiceLoss = GeneralizedDiceLoss()
    criterion2 = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.])).to(device)
    global_step = 0
    best_dice = 0.0
    best_masks = torch.Tensor([])
    best_maskprob = torch.Tensor([])
    # 5. Begin training
    for epoch in range(1, epochs+1):
        
        ### train
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                true_cls = batch['label']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    if seg_task:
                        masks_pred, _ = net(images)
                        if net.n_classes > 1:
                            mask_loss = criterion(masks_pred, true_masks) \
                                + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                            # mask_loss = weightedbceloss(masks_pred, true_masks) \
                            #      + dice_loss(F.softmax(masks_pred, dim=1).float(),
                            #            F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                            #            multiclass=True)
                        else:
                            ##sigmoid归一化到0-1
                            if net.out_sigmoid:
                                mask_loss = dice_loss(masks_pred.squeeze().float(), true_masks.float(), multiclass=False)
                            else:
                                mask_loss = dice_loss(torch.sigmoid(masks_pred.squeeze()).float(), true_masks.float(), multiclass=False)
                                            
                    else: 
                        mask_loss = 0       

                    if cls_task:
                        _, cls_pred = net(images)
                        cls_loss = criterion2(cls_pred, true_cls.to(device=device, dtype=torch.long))
                        # cls_loss = Focalloss(cls_pred, F.one_hot(true_cls).to(device=device, dtype=torch.long))
                    else:
                        cls_loss = 0

                    loss = mask_loss + alpha* cls_loss
                loss.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'train mask loss': mask_loss.item() if seg_task else None,
                    'train cls loss': cls_loss.item() if cls_task else None,
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            if seg_task and cls_task:
                logging.info('Train mask loss: {} cls loss: {}'.format(mask_loss.item(), cls_loss.item()))
            elif seg_task and not cls_task:
                logging.info('Train mask loss: {}'.format(mask_loss.item()))
            elif not seg_task and cls_task:
                logging.info('Train cls loss: {}'.format(cls_loss.item()))


        # Evaluation round
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        #####val
        val_dice_score, iou, val_acc, auc, sens, spec, valid_masks, valid_maskprob = undis_evaluate(net, val_loader, device, seg_task, cls_task)
        val_score = val_dice_score + val_acc


        scheduler.step(val_score)
        if seg_task and cls_task:
            logging.info('Validation Dice score: {} iou: {} Acc: {} Auc: {} Sens: {} Spec: {}'.format(val_dice_score, iou, val_acc, auc, sens, spec))
        elif seg_task and not cls_task:
            logging.info('Validation Dice score: {} iou: {}'.format(val_dice_score, iou))
        elif not seg_task and cls_task:
            logging.info('Validation Acc: {} Auc: {} Sens: {} Spec: {}'.format(val_acc, auc, sens, spec))

        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'val_loss': val_score,
            'validation Dice': val_dice_score if seg_task else None,
            'iou': iou if iou else None,
            'validation acc': val_acc if cls_task else None,
            'auc': auc if auc else None,
            'senstivity': sens if cls_task else None,
            'specificity': spec if cls_task else None,
            'images': wandb.Image(images[0].cpu()),
            'masks': {
                'true': wandb.Image(true_masks[0].float().cpu()),
                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
            } if seg_task else None,
            'step': global_step,
            'epoch': epoch,
            **histograms
        })

        if best_dice < val_dice_score:
            best_dice = val_dice_score
            best_masks = valid_masks
            best_maskprob = valid_maskprob
            if save_checkpoint:
                torch.save(net.state_dict(), 'best_dice.pth')



        # if save_checkpoint:

        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')
    savedir = 'results/'   
    for i in range(192):
        # print(i)
        img = val_set[i]['image'][0].numpy()
        mask = val_set[i]['mask'].numpy()
        mask = mask *255
        ###创建文件夹
        import matplotlib.pyplot as plt
        plt.imsave(savedir + 'image{}.png'.format(i), img, cmap='gray', format='png')
        plt.imsave(savedir + 'gt{}.png'.format(i), mask*255, cmap='gray')
        plt.imsave(savedir + 'pred{}.png'.format(i), best_masks[i]*255, cmap='gray')
        plt.imsave(savedir + 'probmap{}.png'.format(i), best_maskprob[i]*255, cmap='gray')


    # 删除临时缓存文件
    time_str = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    # if os.path.exists(checkpoint_path) is True:
    #     os.remove(checkpoint_path)
    torch.save(net.state_dict(), 'MODEL_' + time_str + '.pth')

    # cleanup()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel



#####是否开启多任务
    seg_task = True
    cls_task = False
    alpha = 0.8
####网络输入通道数
    in_channels = 1
    net_name = 'ResUNet'
###自定义网络
    if net_name == 'Resnet18':
        net = ResNet18(n_channels=in_channels, n_classes=2)
    elif net_name == 'MRCNet':
        net = MRCNet(n_channels=in_channels, 
                    n_classes=args.classes, 
                    bilinear=args.bilinear, 
                    seg_task=seg_task, 
                    cls_task=cls_task
                )
        logging.info(f'Network:\n'
            f'\t{net.n_channels} input channels\n'
            f'\t{net.n_classes} output channels (classes)\n'
            f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    elif net_name == 'ResUNet':
        net = mtihead_ResUnet(n_channels=in_channels, 
                            n_classes=args.classes, 
                            bilinear=args.bilinear, 
                            seg_task=seg_task, 
                            cls_task=cls_task
                        )
        logging.info(f'Network:\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
        # net = ResUnet(n_channels=in_channels, n_classes=2)
    elif net_name == 'ResUnetPlusPlus':
        net = ResUnetPlusPlus(n_channels=in_channels, n_classes=2)
    elif net_name == 'UNet':
        net = mtihead_Unet(n_channels=in_channels, 
                            n_classes=args.classes, 
                            bilinear=args.bilinear, 
                            seg_task=seg_task, 
                            cls_task=cls_task
                        )
        logging.info(f'Network:\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  input_channel=in_channels,
                  seg_task=seg_task,
                  cls_task=cls_task
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise


