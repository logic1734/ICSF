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
from Modules import Codec, Filter, Fusion
from parameter import Parameter
from img_read_save import *
from utils import H5Dataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DataPath_reged=[r"data_train/data/MSFEMFM_pu_i128_s200.h5"]
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
model_fusion = nn.DataParallel(Fusion(num_filters=par.num_filters)).to(device)

# Optimize functions and adjustments
optimizer = torch.optim.Adam(model_codec.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)
scaler = torch.cuda.amp.GradScaler()

# Loss functions
criteria_fusion = Fusionloss(mu=par.mu2).to(device)

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
    for i, (Ix, Iy, _) in enumerate(ld):
        Ix, Iy = Ix.to(device), Iy.to(device)
        model_codec.zero_grad()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            fx = model_codec.forward(Ix, mode='encode')
            fy = model_codec.forward(Iy, mode='encode')
            hfx, ifx = model_filter(fx)
            hfy, ify = model_filter(fy)
            Ifus = model_codec.forward(model_fusion(torch.cat([hfx, hfy, ifx, ify],dim=1)), mode = 'decode')
            # loss
            FuLoss, lossin, lossg= criteria_fusion(Ix, Iy, Ifus)
            loss = FuLoss
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
        tb.add_scalar('lossin', lossin.item(), batches_done)
        tb.add_scalar('lossg', lossg.item(), batches_done)
        tb.add_scalar('fu_loss', loss.item(), batches_done)
    # update learning rate
    scheduler.step()
    if optimizer.param_groups[0]['lr'] <= 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6
# Save model
if True:
    checkpoint = {
        'Fusion': model_filter.state_dict()
        }
    file_name = 'ICSF_U'+ str(patch_size[0]) + '_e' + str(epoch_num) +'.pth'
    save_path = os.path.join('./pth_save/', file_name)
    torch.save(checkpoint, save_path)
    print('\nThe model has been saved in:' + str(save_path))