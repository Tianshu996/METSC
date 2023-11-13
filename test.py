#!/usr/bin/python3.8
# @Author  : Tianshu Zheng
# @Email   : zhengtianshu996@gmail.com
# @File    : test.py
# @Software: PyCharm

'''
This is the Inference code of our publication in Medical Image Analysis (T. Zheng et al. “A microstructure estimation Transformer inspired by sparse representation for diffusion MRI,” Med. Image Anal., vol. 86, p. 102788, 2023)
'''

import argparse
from torch.autograd import Variable
import torch
import numpy as np
from net import SparseReconstruction
from net import Mapping
from vit_pytorch import ViT
from Dataform import Mydataset
import math
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=25230, help='size of the batches')
parser.add_argument('--dataset', type=str, default='./example/data.mat', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=60, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', default='True',help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use')
parser.add_argument('--net_sp', type=str, default='./model/sp.pth')
parser.add_argument('--Mapping', type=str, default='./model/mapping.pth')
parser.add_argument('--v', type=str, default='./model/v.pth')
parser.add_argument('--size1', type=int, default=3, help='size of the data crop (squared assumed)')
parser.add_argument('--mask', type=str, default='./example/mask.nii',help='mask used for generation')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
net_sp = SparseReconstruction()

v = ViT(
    image_size=3,
    patch_size=1,
    num_classes=60 * 1 * 1,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1
)

Mapping = Mapping()


if opt.cuda:
    net_sp.cuda()
    v.cuda()
    Mapping.cuda()

# Load state dicts

net_sp.load_state_dict(torch.load(opt.net_sp))
Mapping.load_state_dict(torch.load(opt.Mapping))
v.load_state_dict(torch.load(opt.v))

# Set model's test mode
net_sp.eval()
Mapping.eval()
v.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size1, opt.size1)
input_B = Tensor(opt.batchSize, opt.output_nc, 1)

tau = Tensor(1, 1).fill_(1e-10)

# Dataset loader
testset = Mydataset(opt.dataset)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=opt.batchSize,
    shuffle=False,
)

###### Testing######

store = []

with torch.no_grad():
    for i, batch in enumerate(testloader):
        # Set model input
        real_A = (Variable(input_A.copy_(batch['A'])))

        real_B = Variable(input_B.copy_(batch['B']))

        t = v(real_A)

        t = t.reshape(real_A.shape[0], real_A.shape[1], 1, 1)

        dataconv = t + (real_A[:, :, 1, 1]).reshape(real_A.shape[0], real_A.shape[1], 1, 1)

        same_B = net_sp(dataconv)

        same_B = same_B.reshape(same_B.shape[0], same_B.shape[1])

        Viso = (same_B[:, -1]).reshape(same_B.shape[0], 1)

        Vf = (same_B[:, :-1]).reshape(same_B.shape[0], -1)

        Vf = (Vf + tau) / torch.norm((Vf + tau), p=1, dim=1).reshape(Vf.shape[0], 1)

        Vf = Vf.unsqueeze(2).unsqueeze(3)

        sameVf = Mapping(Vf)

        Vic = (sameVf[:, 0, :, :]).squeeze(2)

        ODI = (sameVf[:, 1, :, :]).squeeze(2)

        same_B4 = torch.cat([Vic, Viso, ODI], dim=1)

        output = same_B4.detach().cpu().numpy()

        result1 = np.array(output)

        store.append(result1)

        result1 = np.array(store)

        result1 = np.reshape(result1, (result1.shape[0] * result1.shape[1], 3))


    mask_nii = nib.load(opt.mask)

    mask = mask_nii.get_fdata()

    affine = mask_nii.affine

    for ii in range(3):

        data = result1[:,ii].reshape(mask.shape,order='F') * mask

        image = nib.Nifti1Image(data, affine)

        nib.save(image,'./output/result'+str(ii)+'.nii.gz')

