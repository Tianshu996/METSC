#!/usr/bin/python3.8
# @Author  : Tianshu Zheng
# @Email   : zhengtianshu996@gmail.com
# @File    : test.py
# @Software: PyCharm

'''
This is the network of our publication in Medical Image Analysis (T. Zheng et al. “A microstructure estimation Transformer inspired by sparse representation for diffusion MRI,” Med. Image Anal., vol. 86, p. 102788, 2023)
'''

import torch
import torch.nn as nn



class Dictionary_Block(nn.Module):
    def __init__(self,input):
        super(Dictionary_Block, self).__init__()

        Dict_block= [

            nn.Threshold(0.001, 0, inplace=True),

            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, stride=1, bias=True),

        ]

        self.Dict_block = nn.Sequential(*Dict_block)

    def forward(self, x):

        return self.Dict_block(x)

class W_layer(nn.Module):
    def __init__(self,input):
        super(W_layer, self).__init__()

        W_block = [

            nn.Conv2d(in_channels=60, out_channels=input, kernel_size=1, stride=1, bias=True),

        ]

        self.W_block = nn.Sequential(*W_block)


    def forward(self, x):

        return self.W_block(x)

class SparseReconstruction(nn.Module):
    def __init__(self):
        super(SparseReconstruction, self).__init__()

        input = 601

        self.dcblock1 = nn.Sequential(

            Dictionary_Block(input)

        )

        self.wblock = nn.Sequential(

            W_layer(input)

        )

        self.activ = nn.Sequential(

            nn.Threshold(0.001, 0, inplace=True),
        )


    def forward(self,x):

        x1 = self.wblock(x)

        y1 = self.dcblock1(x1)

        y1 = x1 + y1

        for i in range (8):

            y1 = x1 + self.dcblock1(y1)

        y1 = self.activ(y1)

        return y1



class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()

        model = [

            nn.Threshold(0.0001, 0, inplace=False),

            nn.Conv2d(in_channels=600,out_channels=1,kernel_size=1),



        ]

        model1 = [

            nn.Threshold(0.0001, 0, inplace=False),

            nn.Conv2d(in_channels=600,out_channels=1,kernel_size=1),

        ]

        self.model = nn.Sequential(*model)

        self.model1 = nn.Sequential(*model1)

    def forward(self, x):

        output1 = self.model(x)

        output2 = self.model1(x)

        output = torch.cat((output1, output2), dim=1)
        
        return output
