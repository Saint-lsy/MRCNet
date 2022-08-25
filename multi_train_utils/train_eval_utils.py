import sys

from tqdm import tqdm
import torch
import torch.nn as nn
from multi_train_utils.distributed_utils import reduce_value, is_main_process
from utils.dice_score import dice_loss
import torch.nn.functional as F
import wandb
from utils.dice_score import multiclass_dice_coeff, dice_coeff

def train_one_epoch(model, optimizer, data_loader, device, epoch, amp, global_step, experiment):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    # optimizer.zero_grad()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        images = data['image']
        true_masks = data['mask']

        assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=amp):
            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks) \
                    + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, model.module.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)

        # pred = model(images.to(device))


        # loss = loss_function(pred, labels.to(device))
        # loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # optimizer.step()
        # optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if is_main_process():
            global_step += 1
            experiment.log({
                'train loss': mean_loss.item(),
                'step': global_step,
                'epoch': epoch
            })

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device, global_step, experiment, epoch):
    model.eval()
    num_val_batches = len(data_loader)
    dice_score = 0
    # 用于存储预测正确的样本个数
    # sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        image, mask_true = data['image'], data['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, model.module.n_classes).permute(0, 3, 1, 2).float()
        batch_loss = 0
        with torch.no_grad():
            # predict the mask
            mask_pred = model(image)

            # convert to one-hot format
            if model.module.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                batch_loss = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                dice_score += batch_loss
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.module.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                batch_loss = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                dice_score += batch_loss

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    dice_score = reduce_value(dice_score, average=True)

    if is_main_process():
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        experiment.log({
            # 'learning rate': optimizer.param_groups[0]['lr'],
            'validation Dice': dice_score / num_val_batches,
            'images': wandb.Image(image[0].cpu()),
            'masks': {
                'true': wandb.Image(mask_true[0].float().cpu()),
                'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch,
            **histograms
        })    
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

