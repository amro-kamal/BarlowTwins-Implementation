from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import os
import logging
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from earlyStopping import EarlyStopping
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

#TODO
cfg={
'criterion' : torch.nn.CrossEntropyLoss(),
'val_period' : 5,
'epochs' : 200,
'batch_size': 512,
'accumulation_steps' : 1,
'ckpt_path' : '' ,
'load_model' : False, 
'load_path' : '../gdrive/MyDrive/BarlowTwins/checkpoint/renet18/selfsupervised/resnet.pth',
'min_val_acc_to_save' : 30.0,
'gpu' : True,
'min_val_acc_to_save':0,
'early_stopping':False,
}

def predict(model, test_loader,device):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.to(device)
    model.eval()
    preds=[]
    labels=[]
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for images, targets in tqdm(test_loader):
            # Forward pass.
            output = model(images.to(device)) #[bs x out_dim]
            # print(output.shape)
            # Get the label corresponding to the highest predicted probability.
            # print(output.argmax(dim=1, keepdim=True).shape)
            preds+= (output.cpu()) #[bs x 1]
            # print(f'output.shape {output.shape}')
            labels+=targets
            # print('preds',torch.tensor(preds).shape)
    #TODO
    #convert to list
    # for i,p in enumerate(preds):
    #   preds[i]=preds[i].item()
    # print(f'preds [0] shape {preds[0]}')
    # return torch.tensor(preds,dtype=torch.float32) , torch.tensor(labels,dtype=torch.float32)
    return preds , labels
class LC_Dataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (list): list of input fetaures.
            labels (list): list of labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features=features
        self.labels=labels

    def __len__(self):
        return len(self.features)
    #TODO
    def __getitem__(self, idx):
        return ( self.features[idx],self.labels[idx] )

def cifar10_loader(batch_size):
    train_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(32) ,
                                        transforms.ToTensor()  ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                        transforms.RandomHorizontalFlip()
    ])

    val_transforms = transforms.Compose([
                                        transforms.Resize(32) ,
                                        transforms.ToTensor()  ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    train_data = torchvision.datasets.CIFAR10('data/train',train=True,download=True, transform=train_transforms)
    val_data = torchvision.datasets.CIFAR10('data/val',train=False,download=True, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
def train_linear_classifer(cfg):
    """
       Simple training loop for PyTorch linear_classifer.
       cfg: criterion, optimizer ,epochs , model_path='linear_classifer.ckpt' , scheduler=None  ,load_model=False, min_val_acc_to_save=88.0
    """ 
    if cfg['gpu']:
      device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone_model = torchvision.models.resnet18(pretrained=False)
    print('Loading the backbone model from ckpt.....')
    state_dict=torch.load(cfg['load_path'])
    missing_keys, unexpected_keys = backbone_model.load_state_dict(state_dict, strict=False)
    # print(f'missing keys {missing_keys} , unexpected_keys {unexpected_keys}')
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    backbone_model.fc=nn.Identity() #output features shape [bs x 2048]
    print('The backbone model is ready!')

    print('Getting the feartures from resnet')
    #get cifar10 dataset
    cifar10_train_loader, cifar10_val_loader=cifar10_loader(batch_size=cfg['batch_size'])
    #get the 2048-dim features from the model
    train_features, train_labels = predict(backbone_model , cifar10_train_loader ,device)
    # print(f'train_features shape {train_features.shape}')
    LC_train_dataset = LC_Dataset(train_features, train_labels)
    LC_train_loader = DataLoader(LC_train_dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=2)
   
    val_features, val_labels=predict(backbone_model,cifar10_val_loader,device)
    LC_val_dataset=LC_Dataset(val_features, val_labels)
    LC_val_loader=DataLoader(LC_val_dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=2)
    del backbone_model

    print(f'train data {len(train_features)} examples')
    print(f'val data {len(val_features)} examples')

    linear_classifer=nn.Sequential(nn.Linear(512,10))
    cfg['optimizer'] = torch.optim.SGD(linear_classifer.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer=torch.optim.SGD(linear_classifer.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    # cfg['scheduler'] = torch.optim.lr_scheduler.StepLR(cfg['optimizer'], step_size=3, gamma=0.1),
    cfg['scheduler'] = None
    best_val_acc=0
    linear_classifer.train()

    # Move linear_classifer to the device (CPU or GPU).
    #TODO: check if we need to remove the backbone model from the GPU
    linear_classifer.to(device)
    
    # Exponential moving average of the loss.
    ema_loss = None
    losses=[]
    train_accs=[]
    val_accs=[]
    writer=SummaryWriter(os.path.join(cfg['ckpt_path']+'tensorboard'))
    # early_stopping = EarlyStopping(patience=5, verbose=True , path=os.path.join(cfg['ckpt_path']+'/checkpoint.ckpt'),min_val_acc_to_save=cfg['min_val_acc_to_save'] )

    print(f'----- Training on {device} -----')
    # Loop over epochs.
    for epoch in range(cfg['epochs']):
        print(f'epoch {epoch+1}')
        correct = 0
        num_examples=0
        # Loop over data.
        loop=tqdm(enumerate(LC_train_loader , start =epoch*len(LC_train_loader)), total=len(LC_train_loader))
        for step , (inputs, target) in loop:
            # Forward pass.
            # print(f'inputs shape {inputs.shape}')
            output = linear_classifer(inputs.to(device))
            # print(f'output shape {output.shape}')
            loss = cfg['criterion'](output.to(device), target.to(device))
            # print(f'loss {loss.item()}')
            # Backward pass.
            loss = loss / cfg['accumulation_steps'] # Normalize our loss (if averaged)
            # print(f'2 loss {loss}')
            loss.backward()
            # print('loss backward')

            # if epoch+1 % cfg['accumulation_steps']==0:
              # print('optimizer step')
            optimizer.step()
            optimizer.zero_grad()

            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01 
            # Compute the correct classifications
            preds = output.argmax(dim=1, keepdim=True)
            correct+= preds.cpu().eq(target.view_as(preds)).sum().item()
            # print(f'correct {correct}')
            num_examples+= inputs.shape[0]
            train_acc=correct/num_examples
            #tqdm
            loop.set_description(f"Epoch [{epoch+1}/{cfg['epochs']}]")
            loop.set_postfix(loss=ema_loss, acc=train_acc)

        #write the loss to tensorboard    
        writer.add_scalar('train loss', ema_loss, global_step=epoch)
        writer.add_scalar('train acc', train_acc, global_step=epoch)

        losses.append(ema_loss)
        train_accs.append(train_acc)
        #schedular
        if cfg['scheduler']:
          cfg['scheduler'].step()
        #validate
        if (epoch+1) % cfg['val_period'] ==0:
          val_acc = validate(linear_classifer ,LC_val_loader, device)
          #write the loss to tensorboard    
          writer.add_scalar('val acc', val_acc, global_step=epoch)
          val_accs.append(val_acc)
        
        # ckpt_state={'model_sate_dict':model.state_dict(), 'epochs':epoch+1, 'best_val_acc':best_val_acc}
        # if cfg['early_stopping']:
        #   early_stopping(val_acc,chk_point_state)
        #   if early_stopping.early_stop:
        #       print(f'Early stopping at epoch {epoch}/{cfg["epochs"]}....')
        #       break
        print('-------------------------------------------------------------')

    return train_accs , val_accs, losses


def validate(model, data_loader, device):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
    print(f'----- Model Evaluation on {device}-----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        
        # Loop over test data.
        for features, target in data_loader:
          
            # Forward pass.
            output = model(features.to(device))
            
            # Get the label corresponding to the highest predicted probability.
            preds = output.argmax(dim=1, keepdim=True) #[bs x 1]
            
            # Count number of correct predictions.
            correct += preds.cpu().eq(target.view_as(preds)).sum().item()
    model.train()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    print(f'validation accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)')
    return percent
    

if __name__=='__main__':
    train_linear_classifer(cfg)