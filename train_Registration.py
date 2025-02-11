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
from Modules import Codec, Filter, Registration
from parameter import Parameter
from img_read_save import *
from MutualInformation import MutualInformation

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DataPath_reged=[r"data_train/data/MSRSD_pd_i256_s200.h5"]
tb = SummaryWriter(log_dir="./tb_log")
par = Parameter()
codec_pth = r"pth_save/ICSF_C128_e100.pth"
filter_pth = r"pth_save/ICSF_F128_e100.pth"

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
model_codec.load_state_dict(torch.load(codec_pth)['Codec'])
model_codec.eval()
model_filter = nn.DataParallel(Filter(num_filters=par.num_filters)).to(device)
model_filter.load_state_dict(torch.load(filter_pth)['Filter'])
model_filter.eval()
model_reg = nn.DataParallel(Registration(num_filters=par.num_filters)).to(device)

# Optimize functions and adjustments
optimizer = torch.optim.Adam(model_codec.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)
scaler = torch.cuda.amp.GradScaler()

# Loss functions
criteria_L2 = nn.MSELoss().to(device)  
criteria_MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)

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
    model_filter.train() 
    for i, (Ix, Iy, dlt) in enumerate(ld):
        Ix, Iy = Ix.to(device), Iy.to(device)
        model_codec.zero_grad()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            fx = model_codec.forward(Ix, mode='encode')
            fy = model_codec.forward(Iy, mode='encode')
            hfx, ifx = model_filter(fx)
            hfy, ify = model_filter(fy)
            filed, h = model_reg(hfx, hfy,[patch_size[0],patch_size[0]])
            Ihfx, Ihfy= model_codec.forward(hfx, mode='decode'), model_codec.forward(hfy, mode='decode')
            r1Ihfy = torch.nn.functional.grid_sample(Ihfy, filed)
            r2Ihfy = kornia.geometry.transform.warp_perspective(r1Ihfy, h, (r1Ihfy.size(2), r1Ihfy.size(3)))
            # loss
            srloss = criteria_MI(Ihfx, r2Ihfy)
            fploss = criteria_L2(dlt, h)
            loss = srloss + par.mu4 * fploss
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
        tb.add_scalar('srloss', srloss.item(), batches_done)
        tb.add_scalar('fploss', fploss.item(), batches_done)
        tb.add_scalar('re_loss', loss.item(), batches_done)
    # update learning rate
    scheduler.step()
    if optimizer.param_groups[0]['lr'] <= 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6
# Save model
if True:
    checkpoint = {
        'Registration': model_filter.state_dict()
        }
    file_name = 'ICSF_R'+ str(patch_size[0]) + '_e' + str(epoch_num) +'.pth'
    save_path = os.path.join('./pth_save/', file_name)
    torch.save(checkpoint, save_path)
    print('\nThe model has been saved in:' + str(save_path))