import argparse
import logging
from pyexpat import model
import sys
import tempfile
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from unet import UNet
import torch.distributed as dist

from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

from utils.data_loading import LNM_trainDataset, LNM_testDataset

dir_img = Path('/data/lsy/carvana/imgs/train/')
dir_mask = Path('/data/lsy/carvana/masks/train_masks/')
dir_checkpoint = Path('./checkpoints/')



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


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 4,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    ###Distributed
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    
    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    train_loader = DataLoader(train_set,
                            batch_sampler=train_batch_sampler,
                            pin_memory=True,
                            num_workers=nw,)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            pin_memory=True,
                            num_workers=nw)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))
    if rank == 0:
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')
        tb_writer = SummaryWriter()
        experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                    val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                    amp=amp))
    else:
        experiment = 0

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.RMSprop(pg, lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score


    global_step = 0

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    # 5. Begin training
    for epoch in range(1, epochs+1):

        train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(model=net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    amp=amp,
                                    global_step=global_step,
                                    experiment=experiment)


        dice_score = evaluate(model=net,
                            data_loader=val_loader,
                            device=device,
                            global_step=global_step,
                            experiment=experiment,
                            epoch=epoch)
        scheduler.step(dice_score)

        if rank == 0:
            logging.info('Validation Dice score: {}'.format(dice_score))
            # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "Dice score", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], dice_score, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
        torch.save(net.state_dict(), 'MODEL.pth')

    cleanup()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    weights_path = args.load

    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel





    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    if rank ==0:
        logging.info(f'Network:\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    # 如果存在预训练权重则载入
    if weights_path:
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(net.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
