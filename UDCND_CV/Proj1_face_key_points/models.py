## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


# Define normalization type
def define_norm(n_channel,norm_type,n_group=None):
    # define and use different types of normalization steps 
    # Referred to https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html
    if norm_type is 'bn':
        return nn.BatchNorm2d(n_channel)
    elif norm_type is 'gn':
        if n_group is None: n_group=2 # default group num is 2
        return nn.GroupNorm(n_group,n_channel)
    elif norm_type is 'in':
        return nn.GroupNorm(n_channel,n_channel)
    elif norm_type is 'ln':
        return nn.GroupNorm(1,n_channel)
    elif norm_type is None:
        return
    else:
        return ValueError('Normalization type - '+norm_type+' is not defined yet')

class ConvBlock(nn.Module):
    # Conv building block 
    def __init__(self, inplane, outplane, kernel_size=3, stride=1,padding=1,norm_type=None):
        super(ConvBlock,self).__init__()
        # parameters
        self.norm_type = norm_type

        # layers
        self.conv      = nn.Conv2d(inplane,outplane,kernel_size=kernel_size,stride=stride,padding=padding)
        self.norm_layer= define_norm(outplane,norm_type)
        self.relu      = nn.ReLU(inplace=True)
        self.maxpool   = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):

        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class Net(nn.Module):

    def __init__(self,input_dim = 224, n_block=4, n_hidden=None, norm_type=None):
        super(Net, self).__init__()
        
        # initial parameter settings
        self.n_feats = [1, 32, 64, 128, 256]

        if n_hidden is None:
            self.n_hidden = 136*4
        else:
            self.n_hidden = n_hidden
        self.norm_type = norm_type
        
        # Feacture extraction step
        layers = [];
        for ii in range(n_block):
            block = ConvBlock(self.n_feats[ii],self.n_feats[ii+1],norm_type=norm_type)
            layers.append(block)
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2*2*self.n_feats[n_block], self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.n_hidden, 68*2))
        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.contiguous().view(x.shape[0],-1)
        x = self.classifier(x)
        return x


