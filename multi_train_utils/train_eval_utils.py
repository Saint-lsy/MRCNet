import sys

from tqdm import tqdm
import torch
import torch.nn as nn
from multi_train_utils.distributed_utils import reduce_value, is_main_process
from utils.dice_score import dice_loss
import torch.nn.functional as F
import wandb
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from sklearn.metrics import roc_auc_score

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
        true_cls = data['label']
        assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=amp):
            masks_pred, cls_pred = model(images)
            mask_loss = criterion(masks_pred, true_masks) \
                    + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, model.module.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)
            cls_loss = criterion(cls_pred, true_cls.to(device=device, dtype=torch.long))
            loss = mask_loss + cls_loss
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
    val_num = 192
    # 用于存储预测正确的样本个数
    # sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    acc = 0.0
    TP, TN, FP, FN =0, 0, 0, 0
    for step, data in enumerate(data_loader):
        image, mask_true, cls_true = data['image'], data['mask'], data['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, model.module.n_classes).permute(0, 3, 1, 2).float()
        batch_loss = 0

        with torch.no_grad():
            # predict the mask
            mask_pred, cls_pred = model(image)

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

                # compute the classification accuracy
                cls_pred = torch.max(cls_pred, dim=1)[1]
                TP += torch.sum((cls_pred == cls_true.to(device)) & (cls_true.to(device) == 1))
                TN += torch.sum((cls_pred == cls_true.to(device)) & (cls_true.to(device) == 0))
                FP += torch.sum((cls_pred != cls_true.to(device)) & (cls_true.to(device) == 0))
                FN += torch.sum((cls_pred != cls_true.to(device)) & (cls_true.to(device) == 1))
                acc += torch.eq(cls_pred, cls_true.to(device)).sum()
    
 
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    ####对于多进程，需要将所有进程的结果汇总 求和取平均
    dice_score = reduce_value(dice_score, average=True)
    acc = reduce_value(acc, average=False)
    TP = reduce_value(TP, average=False)
    TN = reduce_value(TN, average=False)
    FP = reduce_value(FP, average=False)
    FN = reduce_value(FN, average=False)
    if num_val_batches != 0:
        dice_score = dice_score / num_val_batches
        val_accurate = acc / val_num
        sens = TP.double() / (TP + FN)
        spec = TN.double() / (TN + FP)
    if is_main_process():
        print('validation \n Acc: {:.4f} sens: {:.4f} spec {:.4f}'.format(val_accurate, sens, spec))
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
        experiment.log({
            # 'learning rate': optimizer.param_groups[0]['lr'],
            'validation Dice': dice_score,
            'validation Acc': val_accurate,
            'images': wandb.Image(image[0].cpu()),
            'masks': {
                'true': wandb.Image(mask_true[0].float().cpu()),
                'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch,
            **histograms
        })    
        
    return dice_score, val_accurate



def undis_evaluate(net, dataloader, device, seg_task, cls_task):
    net.eval()
    num_val_batches = len(dataloader)
    val_num = 192
    dice_score = 0
    acc = 0.0
    # auc2 = 0.0
    TP, TN, FP, FN =0, 0, 0, 0
    valid_probs = torch.Tensor([]).to(device)
    valid_masks = torch.Tensor([]).to(device)
    valid_labels = torch.IntTensor([])
    valid_maskprob = torch.IntTensor([]).to(device)

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        image, mask_true, cls_true = batch['image'], batch['mask'], batch['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # net = net.cuda()
        with torch.no_grad():
            # predict the mask
            if seg_task:
                mask_pred, _ = net(image)          
                if net.n_classes == 1:
                    mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred.squeeze(), mask_true.float(), reduce_batch_first=False)
                else:
                    mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    valid_maskprob = torch.cat((valid_maskprob, F.softmax(mask_pred, dim=1)[:,1]), 0)

                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)            
                    valid_masks = torch.cat((valid_masks, mask_pred[:,1]), 0)
                    
            if cls_task:
                # #####计算分类准确率
                _, cls_pred = net(image)
                valid_probs = torch.cat((valid_probs, cls_pred), 0)
                valid_labels = torch.cat((valid_labels, cls_true), 0)
                
                cls_pred = torch.max(cls_pred, dim=1)[1]
                TP += torch.sum((cls_pred == cls_true.to(device)) & (cls_true.to(device) == 1))
                TN += torch.sum((cls_pred == cls_true.to(device)) & (cls_true.to(device) == 0))
                FP += torch.sum((cls_pred != cls_true.to(device)) & (cls_true.to(device) == 0))
                FN += torch.sum((cls_pred != cls_true.to(device)) & (cls_true.to(device) == 1))
                acc += torch.eq(cls_pred, cls_true.to(device)).sum()
                    
    if num_val_batches != 0:
        if seg_task:
            dice_score = dice_score / num_val_batches
            iou = dice_score / (2 - dice_score)
            valid_masks = valid_masks.cpu()
            valid_maskprob = valid_maskprob.cpu()
        if cls_task:
            val_accurate = acc / val_num
            auc1 = roc_auc_score(F.one_hot(valid_labels), valid_probs.cpu(), average='macro')
            auc2 = roc_auc_score(valid_labels, valid_probs[:,0].cpu())
            sens = TP.double() / (TP + FN)
            spec = TN.double() / (TN + FP)           
    net.train()

    # Fixes a potential division by zero error
    if seg_task and cls_task:
        return dice_score, iou, val_accurate, auc2, sens, spec, 0, 0
    elif seg_task:
        return dice_score, iou, 0, 0, 0, 0, valid_masks, valid_maskprob
    elif cls_task:
        return 0, 0, val_accurate, auc2, sens, spec, 0, 0