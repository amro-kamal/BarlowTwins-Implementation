import torch
import torchvision
import torch.nn as nn
import copy
import torch_xla.core.xla_model as xm



class BarlowTwins(nn.Module):
    ''' 
      resnet with custom FC head
    '''
    def __init__(self, args , projection_dims=[512,512,512] ):
        super().__init__()
        self.projection_dims=projection_dims
        self.args=args
        self.resnet_backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.resnet_backbone.fc = nn.Identity()
        fc_dims = [2048]+ self.projection_dims
        layers = []
        for i in range(len(fc_dims)-2):
            layers.append( nn.Linear(fc_dims[i], fc_dims[i+1], bias=False) )
            layers.append(nn.BatchNorm1d(fc_dims[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(fc_dims[-2], fc_dims[-1], bias=False))
        self.fc_prejector = nn.Sequential( *layers ) #[b x d]
        self.normalization = nn.BatchNorm1d(fc_dims[-1], affine=False)


    def forward(self, x1, x2):
        # print('calling model forward')
        z1 = self.normalization(self.fc_prejector(self.resnet_backbone(x1))) #[b x d]
        # print(f'z1 shape {z1.shape}')
        z2 = self.normalization(self.fc_prejector(self.resnet_backbone(x2))) #[b x d]
        # cross_correlation = (z1.T @ z2)/z1.shape[0] #[d x d]

        cross_correlation = z1.T @ z2 #[d x d]
        # print('cross correlation ✅✅✅')
        cross_correlation.div_(z1.shape[0])
        # torch.distributed.all_reduce(cross_correlation)
        # xm.all_reduce(xm.REDUCE_SUM , inputs=cross_correlation)
        # sum the cross-correlation matrix between all gpus
        on_diag = torch.diagonal(cross_correlation).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cross_correlation).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag

    
        return loss

        
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()