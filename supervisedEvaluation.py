import torchvision
import torchvision.datasets as datasets 
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from earlyStopping import EarlyStopping
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

parser.add_argument('--pretrained-path', default='../gdrive/MyDrive/BarlowTwins/checkpoint/renet18/selfsupervised/resnet.pth', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('--data',default='evaldata', type=Path, metavar='CIFAR10',
                    help='path to dataset')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.3, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


 #TODO : save ad load the model and optimizer   
args=parser.parse_args()                
def main():
    args.rank=0
    args.seed=42
    
    start_time = time.time()

    print('calling spawn')
    xmp.spawn(XLA_trainer, args=(args,), nprocs=8, start_method='fork')


SERIAL_EXEC = xmp.MpSerialExecutor()
def XLA_trainer(index, args):

    def test_loop_fn(val_loader):
        total_samples = 0
        correct = 0
        para_val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
        for data, target in para_val_loader:
          output = model(data)
          pred = output.max(1, keepdim=True)[1]
          correct += pred.eq(target.view_as(pred)).sum().item()
          total_samples += data.size()[0]

        accuracy = 100.0 * correct / total_samples
        print('[xla:{}] Accuracy={:.2f}%'.format(
            xm.get_ordinal(), accuracy), flush=True)
        return accuracy, data, pred, target

    args.rank = index
    '''
    1-create sampler
    2-create dataloader
    3-call train() function
    '''
  
    torch.manual_seed(args.seed)
    device = xm.xla_device()  
   
    model = torchvision.models.resnet18(pretrained=False)
    state_dict = torch.load(args.pretrained_path, map_location='cpu')

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    model.fc=nn.Linear(512,10)
    # model.fc.weight.data.normal_(mean=0.0, std=0.01)
    # model.fc.bias.data.zero_()
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
    optimizer = optim.SGD(model.parameters(), 0.002, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if False and (args.checkpoint_dir / 'checkpoint.pth').is_file():
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

    #Data loading code
    traindir = args.data / 'train'
    valdir = args.data / 'val'


    def get_data():

      if xm.is_master_ordinal():
        logger.info('Downloading the data.......')
      train_dataset = datasets.CIFAR10(
          "/data/train",
          train=True,
          download=True,
          transform=transforms.Compose([
              transforms.RandomResizedCrop(64),
              transforms.ToTensor(),
              transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
      )
          
      val_dataset = datasets.CIFAR10(
          "/data/val",
          train=False,
          download=True,
              transform=transforms.Compose([
              transforms.Resize(64),
              transforms.CenterCrop(64),
              transforms.ToTensor(),
              transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
          ])          ) 

      if args.weights == 'finetune':
        lengths = [int(len(train_dataset)*0.1), int(len(train_dataset)*0.9)]
        train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths)
      return train_dataset , val_dataset


    train_dataset ,val_dataset =SERIAL_EXEC.run(get_data)

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
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

    early_stopping = EarlyStopping(patience=1000, verbose=True ,path=args.checkpoint_dir/'checkpoint.pth' )
    writer=SummaryWriter(args.checkpoint_dir/'tensorboard')

    start_time = time.time()
    model.to(device)
    for epoch in range(start_epoch, args.epochs):
        epoch_loss=0
        num_examples=0
        correct=0
        if xm.is_master_ordinal():
            logger.info(f'epoch {epoch+1}')
        optimizer = optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        model.train()

        #train
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        for step, (images, target) in enumerate(para_train_loader, start=epoch * len(train_loader)):
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()

            xm.optimizer_step(optimizer)  # Note: barrier=True not needed when using ParallelLoader 
            
            pred = output.argmax(dim=1, keepdim=True) #[bs x 1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += images.shape[0] * loss.item()
            num_examples+=images.shape[0]


        epoch_loss=epoch_loss/num_examples
        train_acc=correct/num_examples
################################
        #validation
        model.eval()
        if xm.is_master_ordinal():
          writer.add_scalar('tr_loss', epoch_loss ,global_step=epoch)
          writer.add_scalar('tr_acc', train_acc ,global_step=epoch)
          logger.info(f'----- Model Evaluation on {device}-----')
        
        num_examples = 0
        correct = 0
        para_val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
        for data, target in para_val_loader:
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True) #[bs x 1]
          # print(f'pred shape {pred.shape} , target shape {target.shape} , pred=target {pred.eq(target.view_as(pred)).sum().item()}')

          correct += pred.eq(target.view_as(pred)).sum().item()
          num_examples += data.size()[0]
        val_accuracy = 100.0 * correct / num_examples
        if xm.is_master_ordinal():
          writer.add_scalar('val_acc', val_accuracy, global_step=epoch)
          state = dict(epoch=epoch, model=model.state_dict(),optimizer=optimizer.state_dict())
          early_stopping(val_accuracy,state)

        print(f'[xla:{xm.get_ordinal()}] correct {correct}/{num_examples}, loss {format(epoch_loss,".2f")} val accuracy={format(val_accuracy,".2f")} time {format(time.time()-start_time,".2f")}')
        model.train()

        if early_stopping.early_stop:
            logger.info("Early stopping....")
            break

  ################################

        # sanity check
        # if args.weights == 'freeze':
        #     reference_state_dict = torch.load(args.cnn_path, map_location='cpu')
        #     model_state_dict = model.module.state_dict()
        #     for k in reference_state_dict:
        #         assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k

        # scheduler.step()
        # if args.rank == 0:
        #     state = dict(
        #         epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
        #         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        #     torch.save(state, args.checkpoint_dir / 'finetuned_resnet.pth')



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