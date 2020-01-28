import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import math
import time
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import data_parallel

class _Residual_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()
        
        midc=int(outc*scale)
        

        self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=2, padding=0, groups=1, bias=False)

          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=4, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=4, stride=2, padding=2, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)

        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output


class _Residual_BlockUp(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_BlockUp, self).__init__()

        midc = int(outc * scale)


        self.conv_expand = nn.ConvTranspose2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=2, padding=0,output_padding=1,
                                         groups=1, bias=False)



        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=4, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels=midc, out_channels=outc, kernel_size=4, stride=2, padding=0, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity_data = self.conv_expand(x)
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


class ConvUp(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ConvUp, self).__init__()

        midc = int(outc * scale)

        self.conv_expand = nn.ConvTranspose2d(in_channels=inc, out_channels=outc, kernel_size=4, stride=2, padding=1,
                                              output_padding=0,
                                              groups=1, bias=False)


        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        output = self.relu1(self.conv_expand(x))
        return output

class Encoder(nn.Module):
    def __init__(self, cdim=1, channels=[8,16, 32,64, 128, 256], image_size=256,use_res_block=True):
        super(Encoder, self).__init__() 

        #assert (2 ** len(channels)) * 4 == image_size

        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(in_channels= cdim,out_channels=cc,kernel_size= 4,stride= 2,padding= 1, bias=False),
                #nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
                              )
              
        sz = image_size//2
        for ch in channels[1:]:
            if  use_res_block == True:
                self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            else:
                self.main.add_module('conv_in_{}'.format(sz),nn.Sequential(
                nn.Conv2d(in_channels= cc,out_channels= ch,kernel_size= 4,stride= 2,padding= 1, bias=True),
                     nn.LeakyReLU(0.2),
                              ))
            cc, sz = ch, sz//2
    
    def forward(self, x):

        y = self.main(x) #.view(x.size(0), -1)
        return y

class Decoder(nn.Module):
    def __init__(self, cdim=1, channels=[8,16, 32,64, 128, 256], image_size=256,use_res_block=True):
        super(Decoder, self).__init__() 
        
        #assert (2 ** len(channels)) * 4 == image_size
        
        cc = channels[-1]

        filter_size = int(image_size / (2 ** (len(channels)-1)))

        sz = filter_size

        self.main = nn.Sequential()

        for ch in reversed(channels[:-1]):
            if use_res_block == True:
                self.main.add_module('res_in_{}'.format(sz), _Residual_BlockUp(cc, ch, scale=1.0))
            else:
                self.main.add_module('res_in_{}'.format(sz), ConvUp(cc, ch, scale=1.0))
            cc, sz = ch, sz*2



        self.main.add_module('predict',  nn.ConvTranspose2d(in_channels=cc,out_channels= cdim,kernel_size= 4, stride=2, padding=1,output_padding=0))
                    
    def forward(self, z):

        y = self.main(z)
        s = nn.Sigmoid()
        return s(y)

        

