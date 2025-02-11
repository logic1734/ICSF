import time
import sys
import datetime
import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
from torchmetrics.regression import ConcordanceCorrCoef

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
    def forward(self, tensor):
        eps = torch.finfo(torch.float32).eps
        tensor = F.softmax(tensor.view(tensor.shape[0], -1), dim=1)
        entropy = -torch.sum(tensor * torch.log(tensor + eps), dim=1)
        return torch.mean(entropy)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return sobelx+sobely

class Fusionloss(nn.Module):
    def __init__(self, mu=2):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mu = mu    

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=KF.spatial_gradient(image_y)
        ir_grad=KF.spatial_gradient(image_ir)
        generate_img_grad=KF.spatial_gradient(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+self.mu*loss_grad 
        return loss_total,loss_in,loss_grad
    
class CCLoss(nn.Module):
    def __init__(self):
        super(CCLoss, self).__init__()
        self.criteria_CC = ConcordanceCorrCoef().cuda()

    def forward(self, x, y):            #[batch,1,size,size]
        loss=torch.zeros(x.size(0),dtype=torch.float32)     #[batch]
        for i in range(x.size(0)):
            x1 = torch.flatten(x[i])
            y1 = torch.flatten(y[i])
            ls = self.criteria_CC(x1, y1)
            loss[i] = ls
        return torch.mean(loss)

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        Dlt = np.array(h5f['deform'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR), torch.Tensor(Dlt)
