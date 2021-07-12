from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tqdm
from tqdm.notebook import tqdm
import os

cfg={
'criterion' : torch.nn.CrossEntropyLoss(),
'optimizer' : optimizer,
'scheduler' : torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
'val_period' : 1,
'epochs' : 5,
'accumulation_steps' : 1,
'batch_size'=64,
'ckpt_path' : 'model.ckpt' ,
'load_model' : True, 
'load_path' : 'model.ckpt',
'min_val_acc_to_save' : 30.0,
'gpu' : True
}

def predict(model, test_loader):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
            preds+= (output.argmax(dim=1, keepdim=True).cpu()) #[bs x 1]
            labels+=targets
            # print('preds',torch.tensor(preds).shape)
            
            # Count number of correct predictions.
    #convert to list
    for i,p in enumerate(preds):
      preds[i]=preds[i].item()

    return preds , labels

class LC_Dataset(Dataset):
    """Face Landmarks dataset."""

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

    def __getitem__(self, idx):
        return ( features[idx],labels[idx] )


def train_and_validate(model, train_loader , val_loader, cfg):
    """
       Simple training loop for PyTorch models.
       cfg: criterion, optimizer ,epochs , model_path='model.ckpt' , scheduler=None  ,load_model=False, min_val_acc_to_save=88.0

    """ 
    if cfg['gpu']:
      device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    best_val_acc=0
    # Make sure model is in training mode.
    if cfg['load_model'] and cfg['load_path']:
      print('Loading the model from ckpt.....')
      train_ckpt=torch.load(cfg['load_path'])
      model.load_state_dict(train_ckpt['model'])
      print('The model is ready!')


    model.train()
    cfg['optimizer'].zero_grad()

    # Move model to the device (CPU or GPU).
    model.to(device)
    
    # Exponential moving average of the loss.
    ema_loss = None
    losses=[]
    train_accs=[]
    val_accs=[]
    writer=SummaryWriter('tensorboard')

    print(f'----- Training on {device} -----')
    # Loop over epochs.
    for epoch in range(cfg['epochs']):
        correct = 0
        num_examples=0
        # Loop over data.
        loop=tqdm(enumerate(train_loader , start =epoch*len(train_loader)), total=len(train_loader))
        for step , (images, target) in loop:
            # Forward pass.
            output = model(images.to(device))
            loss = cfg['criterion'](output.to(device), target.to(device))

            # Backward pass.
            loss = loss / cfg['accumulation_steps'] # Normalize our loss (if averaged)
            loss.backward()
            if epoch+1 % cfg['accumulation_steps']==0:
              cfg['optimizer'].step()
              cfg['optimizer'].zero_grad()


            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01 

            # Compute the correct classifications
            preds = output.argmax(dim=1, keepdim=True)
            correct+= preds.cpu().eq(target.view_as(preds)).sum().item()
            num_examples+= images.shape[0]
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
        if epoch+1 % cfg['val_period']==0:
          val_acc = validate(model ,val_loader, device)
          #write the loss to tensorboard    
          writer.add_scalar('val acc', val_acc, global_step=epoch)

          val_accs.append(val_acc)
          if val_acc > best_val_acc and val_acc > cfg['min_val_acc_to_save']:
              print(f'validation accuracy increased from {best_val_acc} to {val_acc}  , saving the model ....')
              #saving training ckpt
              chk_point={'model_sate_dict':model.state_dict(), 'epochs':epoch+1, 'best_val_acc':best_val_acc}
              torch.save(chk_point, cfg['ckpt_path'])
              best_val_acc=val_acc
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
    val_acc = 100. * correct / len(data_loader.sampler)
    print(f'validation accuracy: {correct} / {len(data_loader.sampler)} ({val_acc:.0f}%)')
    return val_acc