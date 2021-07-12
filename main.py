import torchvision
import torchvision.datasets as datasets 
import torch
from torch import nn, optim
from barlowTwins import BarlowTwins
from dataAugmentation import Transform
import torch_xla.distributed.parallel_loader as pl
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import argparse
from pathlib import Path
import logging
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from utils import init_logger
from earlyStopping import EarlyStopping
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import math
from torchsummary import summary

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--data', type=str, default='CIFAR10',
                    help='dataset name')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=1024, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--optimizer', default='SGD', type=str, choices=('SGD', 'LARS'))
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--load-model',default=False, type=bool)
parser.add_argument('--print-model-summary', default=False, type=bool)

logger=init_logger()
args=parser.parse_args()

 #TODO : save ad load the model and optimizer                   
def main():
    '''
     args={'model':model, 'epochs':epochs, 'batch_size':batch_size, 'num_workers':num_workers,
           'lambd':lambd, 'optimizer':optimizer, 'transforms':Transform(), 'seed':seed}
    '''
    SEED=44
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # xm.set_rng_seed(SEED)

    logger.info('building Resnet twins.....')
    model = BarlowTwins(args)
    if args.print_model_summary:
      summary(model, [(3, 32, 32),(3,32,32)])


    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
 
    start_epoch=0
    args.optimizer_state=None
    logger.info(args.load_model)
    # if True and (args.checkpoint_dir / 'checkpoint.pth').is_file():
    #     logger.info(f'loading the model to continue training.....')
    #     ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
    #                       map_location='cpu')
    #     start_epoch = ckpt['epoch']
    #     model.load_state_dict(ckpt['model'])
    #     args.optimizer_state=ckpt['optimizer']

    args.continue_from=start_epoch
    args.model=model
    args.transforms=Transform()
    args.seed=44
 
    logger.info(f'calling spawn ...')
    xmp.spawn(XLA_trainer, args=(args,), nprocs=8, start_method='fork')

SERIAL_EXEC = xmp.MpSerialExecutor()

def XLA_trainer(index, args):
    '''
    1-create sampler
    2-create dataloader
    3-call train() function
    '''

    # if not xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')
    
    # if xm.is_master_ordinal():
    #   logger.info(f'waiting for {device} to download the data')
    def get_data():
      if xm.is_master_ordinal():
        logger.info(f'Downloading the data.......')
      train_dataset = datasets.CIFAR10(
          "/data",
          train=True,
          download=True,
          transform=args.transforms
          ) 
      return train_dataset

    train_dataset=SERIAL_EXEC.run(get_data)
    # if  xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')


    device = xm.xla_device()  
    # Sets a common random seed - both for initialization and ensuring graph is the same
    torch.manual_seed(args.seed)

    # Creates the (distributed) train sampler, which let this process only access
    # its portion of the training dataset.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    logger.info(f'batch size {args.batch_size}')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=True)
    
    train(args.model.to(device), args.epochs, train_loader, args.lambd ,device)


def train(model, epochs, train_loader, lambd, device):
    global_step=0
    writer=SummaryWriter(args.checkpoint_dir/'tensorboard')
    start_time = time.time()
    early_stopping = EarlyStopping(patience=5, verbose=True ,path=args.checkpoint_dir/'checkpoint.pth' )
    if args.optimizer=='SGD':
      optimizer = optim.SGD(model.parameters(), lr=1e-3,
                              momentum=0.9, weight_decay=5e-4)
    else: 
      param_weights = []
      param_biases = []
      for param in model.parameters():
          if param.ndim == 1:
              param_biases.append(param)
          else:
              param_weights.append(param)
 
      parameters = [{'params': param_weights}, {'params': param_biases}]                       
      optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                    weight_decay_filter=exclude_bias_and_norm,
                    lars_adaptation_filter=exclude_bias_and_norm)

    #for continuing training
    if args.optimizer_state:
        #The loaded optimizer must be compatible with the choice in args.optimizer
        optimizer.load_state_dict(args.optimizer)

    for epoch in range(args.continue_from, args.continue_from+epochs):
        if xm.is_master_ordinal():
            logger.info(f'[{xm.get_ordinal()}] device {device} , epoch {epoch+1}')
        epoch_loss=0
        num_examples=0
        epoch_start_time = time.time()
        model.train()

        #ParallelLoader, so each TPU core has unique batch
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        step = epoch * len(para_train_loader)-1
        for ((x1, x2), _) in para_train_loader: #, start=epoch * len(para_train_loader)):
            step+=1
            optimizer.zero_grad()
            loss=model(x1, x2)
            if xm.is_master_ordinal:
              writer.add_scalar('ssloss',loss.item(),global_step=global_step)
              global_step+=1
            if args.optimizer=='LARS':
              lr_schedular(args, optimizer, para_train_loader, step)
            loss.backward()
            epoch_loss += x1.shape[0] * loss.item()
            num_examples+=x1.shape[0]

            xm.optimizer_step(optimizer)
            # if xm.is_master_ordinal():
            #    print('sum ',torch.sum(model.state_dict()['resnet_backbone.conv1.weight']))
        
        # if xm.is_master_ordinal():
            # if xm.is_master_ordinal():
            #    print('sum ',torch.sum(model.state_dict()['resnet_backbone.conv1.weight']))
          
       #saving the model if the loss dicreased  
        epoch_loss=epoch_loss/num_examples
        if xm.is_master_ordinal():
           logger.info(f'epoch {epoch+1} ended, loss : {epoch_loss}, time: {int(time.time() - epoch_start_time)}, global steps: {global_step}')
           state = dict(epoch=epoch, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
           early_stopping(epoch_loss,state)
        if early_stopping.early_stop:
            logger.info("Early stopping....")
            break

    #save the final model, not necessary the best!!! 
    if xm.is_master_ordinal():
      logger.info(f'saving the final model , loss: {epoch_loss} .....🤪🤪🤪🤪🤪🤪🤪')
      xm.save( list( model.children())[0].state_dict(), args.checkpoint_dir/'resnet.pth', master_only=True, global_master=False)
      logger.info('model saved')



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
    
#LARS optimizer : the code is from the original implementation
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def exclude_bias_and_norm(p):
    return p.ndim == 1

if __name__ == '__main__':
    main()