import torch
import torchvision
import torch.nn as nn
import copy



class BarlowTwins(nn.Module):
    ''' 
      resnet with custom FC head
    '''
    def __init__(self, projection_dims=[2048,2048,2048] ):
        super().__init__()
        self.resnet_backbone = torch.models.resnet50(pretrained=Flase)
        self.resnet_backbone.fc = nn.Identity()
        fc_dims = [2028]+ self.projection_dims
        layers = []
        for i in range(len(fc_dims)-2):
            layers.append( nn.Linear(fc_dims[i],fc_dims[i+1]) )
            layers.append(nn.BatchNorm1d(fc_dims[i+1]))
            layers.append(nn.Relu())
        layers.append(nn.Linear(fc_dims[-2],fc_dims[-2]))
        self.fc_prejector = nn.ModuleList( layers ) #[b x d]
        self.normalization = nn.BatchNorm1d(afffine=False)


    def forward(self, x1, x2, lambd):
        z1 = self.normalization(self.fc_prejector(self.resnet_backbone(x1))) #[b x d]
        z2 = self.normalization(self.fc_prejector(self.resnet_backbone(x2))) #[b x d]
        cross_correlation = (z1.T @ z2)/z1.shape[0] #[d x d]

        on_diag = torch.diagonal(cross_correlation).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cross_correlation).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        return loss

        
