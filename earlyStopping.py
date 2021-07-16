import numpy as np
import torch
import torch_xla.core.xla_model as xm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print,min_val_acc_to_save=0.0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.min_val_acc_to_save=min_val_acc_to_save
    def __call__(self, val_acc, state):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, state)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('hello from early stooping')
            self.best_score = score
            if score>self.min_val_acc_to_save:
              self.save_checkpoint(val_acc, state)
            self.counter = 0

    def save_checkpoint(self, val_acc, state):
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(f'Validation acc decreased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving the model ...')
        # torch.save(model.state_dict(), self.path)
        if xm.is_master_ordinal():
          state['best_val_acc']=self.best_score
          torch.save(state, self.path)#, master_only=True, global_master=False)
        print('model saved ✅✅')

        self.val_acc_max = val_acc