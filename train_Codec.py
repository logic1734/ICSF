# -*- coding: utf-8 -*-
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import datetime
import time
import torch
import torch.nn as nn
import kornia
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from Modules import Codec
from parameter import Parameter
from img_read_save import *
from utils import H5Dataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DataPath_reged=[r"data_train/data/MSFEMFM_pu_i128_s200.h5"]
tb = SummaryWriter(log_dir="./tb_log")
par = Parameter()

# Parameters for training
epoch_num = 100
lr = 1e-4
weight_decay = 1e-6
batch_size = 4
clip_grad_norm_value = 0.001 
optim_step = 20
optim_gamma = 0.8
patch_size = {}
patch_size[0] = int(DataPath_reged[0][DataPath_reged[0].find("_i")+2:DataPath_reged[0].find("_s")])
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
print("GPU_number:", GPU_number)

# Model import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_codec = nn.DataParallel(Codec(num_iter=par.Codec_iter, num_filters=par.num_filters)).to(device)

# Optimize functions and adjustments
optimizer = torch.optim.Adam(model_codec.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)
scaler = torch.cuda.amp.GradScaler()

# Loss functions
criteria_L2 = nn.MSELoss().to(device)  
criteria_SSIM = kornia.losses.SSIMLoss(11, reduction='mean').to(device)

# Data Loader
trainloader_r = DataLoader(H5Dataset(DataPath_reged[0]), batch_size=batch_size, shuffle=True, num_workers=0)
loader = {'train_fu': trainloader_r}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
step = 0
torch.backends.cudnn.benchmark = True
ld = loader['train_fu']
prev_time = time.time()
for epoch in range(epoch_num):
    model_codec.train() 
    for i, (Ix, Iy, _) in enumerate(ld):
        Ix, Iy = Ix.to(device), Iy.to(device)
        model_codec.zero_grad()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            Ixre = model_codec.forward(model_codec.forward(Ix, mode='train'),mode='decode')
            Iyre = model_codec.forward(model_codec.forward(Iy, mode='train'),mode='decode')
            # loss
            L2Loss = criteria_L2(Ix, Ixre) + criteria_L2(Iy, Iyre) 
            SSIMLoss= criteria_SSIM(Ix, Ixre) +criteria_SSIM(Iy, Iyre)
            loss = L2Loss + par.mu1 * SSIMLoss
        loss.backward()
        nn.utils.clip_grad_norm_(model_codec.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer.step()
        # time
        batches_all = epoch_num * len(ld)
        batches_done = epoch*len(ld)+i
        batches_left = batches_all - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s GPUM: %d"
                % (
                    epoch+1,
                    epoch_num,
                    i+1,
                    len(ld),
                    loss.item(),
                    time_left,
                    torch.cuda.memory_allocated(device),
                ))
        tb.add_scalar('L2', L2Loss.item(), batches_done)
        tb.add_scalar('SSIM', SSIMLoss.item(), batches_done)
        tb.add_scalar('reb_loss', loss.item(), batches_done)
    # update learning rate
    scheduler.step()
    if optimizer.param_groups[0]['lr'] <= 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6
# Save model
if True:
    checkpoint = {
        'Codec': model_codec.state_dict()
        }
    file_name = 'ICSF_C'+ str(patch_size[0]) + '_e' + str(epoch_num) +'.pth'
    save_path = os.path.join('./pth_save/', file_name)
    torch.save(checkpoint, save_path)
    print('\nThe model has been saved in:' + str(save_path))