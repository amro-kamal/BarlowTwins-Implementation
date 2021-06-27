import torchvision
import torchvision.datasets as datasets 
import torch
from torch import nn, optim
import torch_xla.distributed.parallel_loader as pl
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import argparse
from pathlib import Path
from utils import init_logger
import json
import os
import random
import signal
import sys
import time
import urllib
logger = init_logger('eval.log')


parser = argparse.ArgumentParser(description='Barlow Twins Training')
# parser.add_argument('data', type=str, metavar='CIFAR10',
#                     help='path to dataset')
parser.add_argument('pretrained', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.3, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/resnet50_finetuned/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


 #TODO : save ad load the model and optimizer   
args=parser.parse_args()                
def main():
    args.rank=0
    
    start_time = time.time()

    print('calling spawn')
    xmp.spawn(XLA_trainer, args=(args,), nprocs=8, start_method='fork')

def XLA_trainer(index, args):
    args.rank = index
    '''
    1-create sampler
    2-create dataloader
    3-call train() function
    '''
    print('starting xla traininer')
    # Sets a common random seed - both for initialization and ensuring graph is the same
    # torch.manual_seed(args.seed)
    # print('setting seed')
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()  
    logger.info(f'Training will start on {device}')
    print(f'Training will start on {device}')
    # Downloads train and test datasets
    # Note: master goes first and downloads the dataset only once (xm.rendezvous)
    #   all the other workers wait for the master to be done downloading.


    model = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.load(args.pretrained, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f'missing keys {missing_keys} , unexpected_keys {unexpected_keys}')
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    criterion = nn.CrossEntropyLoss()

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    traindir = args.data / 'train'
    valdir = args.data / 'val'

    normalize = transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))


    if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')
    
    logger.info('Downloading the data.......')

    train_dataset = datasets.CIFAR10(traindir,
            train=True, download=True, 
            transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    

    val_dataset = datasets.ImageFolder(valdir, 
            train=True, download=True,
            transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if  xm.is_master_ordinal():
        xm.rendezvous('download_only_once')
    
    print('creating the sampler')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    print('sampler created ✅')

    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    print('creating the dataloader')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,
    drop_last=True)
    print('dataloader is ready ✅')


    # train(model.to(device), args.epochs, train_loader, args.lambd, args.optimizer ,device)

    start_time = time.time()
    model.to(device)
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        # train_sampler.set_epoch(epoch)
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        for step, (images, target) in enumerate(para_train_loader, start=epoch * len(train_loader)):
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)  # Note: barrier=True not needed when using ParallelLoader 
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(len(xm.get_xla_supported_devices())), 0)
                # # if args.rank == 0:
                if xm.is_master_ordinal():
                    pg = optimizer.param_groups
                    lr_classifier = pg[0]['lr']
                    lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                    stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                 lr_classifier=lr_classifier, loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        # evaluate
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter('Acc@1')
            top5 = AverageMeter('Acc@5')

            para_val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images)
                    #batch accuracy
                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

        # sanity check
        if args.weights == 'freeze':
            reference_state_dict = torch.load(args.pretrained, map_location='cpu')
            model_state_dict = model.module.state_dict()
            for k in reference_state_dict:
                assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

# def train(model, epochs, train_loader, lambd, optimizer, device):
#     model.zero_grad()
    
#     for e in range(epochs):
#         print(f'device {device} , epoch {e}')
#         epoch_loss=0
#         model.train()

#         #ParallelLoader, so each TPU core has unique batch
#         para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
#         for step, ((x1, x2), _) in enumerate(para_train_loader, start=e * len(para_train_loader)):
#             print(f'device {device} , epoch {e} , x1 shape {x1.shape} , x2 shape {x2.shape}')
#             loss=model(x1, x2)
#             print('done model forward✅✅✅')
#             lr_schedular(args, optimizer, para_train_loader, step)
#             # epoch_loss+=loss.item()

#             loss.backword()
#             optimizer.step()
#             model.zero_grad()
#             print('done batch ✅✅✅')
#         print(f'epoch {e}: loss= {epoch_loss}')


def lr_schedular(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()