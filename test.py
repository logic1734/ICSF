
import os
import numpy as np
import torch
import torch.nn as nn
import warnings
import logging
import kornia.geometry.transform as kgt

from img_read_save import img_save,image_read_cv2,reso_resize
from Modules import Codec, Filter, Fusion, Registration
from parameter import Parameter
from MutualInformation import MutualInformation

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pth_path=[r"pth_save/ICSF_C128_e100.pth",r"pth_save/ICSF_F128_e100.pth",r"pth_save/ICSF_U128_e100.pth",r"pth_save/ICSF_R128_e100.pth"]
patch_size = int(pth_path[0][pth_path.find("__")+2:pth_path.find("_t")])
par = Parameter()
criteria_MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)
datasets=["test"]
is_develop=False


def output_progress(img, path, color, cvt=False):
    img=(img-torch.min(img))/(torch.max(img)-torch.min(img))
    img = np.squeeze((img * 255).cpu().numpy())
    if cvt:
        img=np.stack((color[:,:,0],color[:,:,1],img),axis=2)
        img=cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img_save(img, img_name.split(sep='.')[0], path)
    return img

for dataset_name in datasets: 
    print("\n"*2+"="*70)
    print("The test result of "+dataset_name+":")
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('test_result_reg',dataset_name)
    test_dev_folder=os.path.join('test_develop_reg',dataset_name)
    # preprogress
    i=0
    for img_name in os.listdir(os.path.join(test_folder,"ir")):
        ir = image_read_cv2(os.path.join(test_folder,"ir", img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(test_folder,"vi", img_name), 'RGB')
        if ir.shape[0] % 16 != 0 or ir.shape[1] % 16 != 0:
            i=i+1 
            ir = reso_resize(ir)
            vi = reso_resize(vi)
            img_save(ir, img_name.split(sep='.')[0], os.path.join(test_folder,"ir"),fmt=img_name.split(sep='.')[-1])
            img_save(vi, img_name.split(sep='.')[0], os.path.join(test_folder,"vi"),fmt=img_name.split(sep='.')[-1])
    print ("preprogress finish: The size of the "+str (i)+" image has been changed")     
    # test
    model_codec = nn.DataParallel(Codec(num_iter=par.Codec_iter, num_filters=par.num_filters)).to(device)
    model_codec.load_state_dict(torch.load(pth_path[0])['Codec'])
    model_codec.eval()
    model_filter = nn.DataParallel(Filter(num_filters=par.num_filters)).to(device)
    model_filter.load_state_dict(torch.load(pth_path[0])['Filter'])
    model_filter.eval()
    model_fusion = nn.DataParallel(Fusion(num_filters=par.num_filters)).to(device)
    model_fusion.load_state_dict(torch.load(pth_path[1])['Fusion'])
    model_fusion.eval()
    model_reg = nn.DataParallel(Registration(num_filters=par.num_filters)).to(device)
    model_reg.load_state_dict(torch.load(pth_path[2])['Registration'])
    model_reg.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):
            Iy = image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')
            pic_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='RGB')
            Ix = cv2.cvtColor(pic_VIS, cv2.COLOR_RGB2HSV)
            data_color=Ix[:,:,0:2]
            Ix = Ix[:,:,2]
            Ix = Ix[np.newaxis,np.newaxis, ...]/255.0
            Iy = Iy[np.newaxis,np.newaxis, ...]/255.0
            H,W = Iy.shape[2],Iy.shape[3]
            Iy,Ix = torch.FloatTensor(Iy),torch.FloatTensor(Ix)
            Ix, Iy = Ix.cuda(), Iy.cuda()
            Mi = criteria_MI(Ix, Iy)
            fx = model_codec.forward(Ix, mode='encode')
            fy = model_codec.forward(Iy, mode='encode')
            hfx, ifx = model_filter(fx)
            hfy, ify = model_filter(fy)
            if abs(Mi) < 0.3:
                print("The mutual information of "+img_name+" is less than ±0.3, the registration is not necessary")
                Ifus = model_codec.forward(model_fusion(torch.cat([hfx, hfy, ifx, ify],dim=1)), mode = 'decode')
            else:
                print("The mutual information of "+img_name+" is greater than ±0.3, the registration is necessary")
                filed, h = model_reg(hfx, hfy,[patch_size[0],patch_size[0]])
                Iyr = kgt.warp_perspective(torch.nn.functional.grid_sample(Iy, filed), h, (H, W))
                hfyr, ifyr = model_filter(model_codec.forward(Iyr, mode='encode'))
                Ifus = model_codec.forward(model_fusion(torch.cat([hfx, hfyr, ifx, ifyr],dim=1)), mode = 'decode')
            # output
            output_progress(Ifus, test_out_folder, data_color, cvt=False)   
        print("="*70)


