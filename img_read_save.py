import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
from skimage.io import imsave
import numpy

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath,fmt='png'):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    image = image.astype(np.uint8)
    imsave(os.path.join(savepath, "{}.".format(imagename) + fmt),image)
    
def reso_resize(x):
    dh=16-(x.shape[0] % 16)
    dw=16-(x.shape[1] % 16)
    x = cv2.resize(x, (x.shape[1]+dw, x.shape[0]+dh))
    return x

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    
def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def Im2Patch(img, win, stride=1, alw=0):# [H, W](0-255)int8
    k = 0
    img = img.astype(np.float32)[None, :, :]/255.
    endc = img.shape[0]
    endh = img.shape[1]
    endw = img.shape[2]
    patch = img[:, alw:endh-win-alw+1:stride, alw:endw-win-alw+1:stride, ]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win, win, TotalPatNum], np.float32)
    location = np.zeros([2, TotalPatNum], np.float32)
    for x in range(alw,endw-win-alw+1,stride):
        for y in range(alw,endh-win-alw+1,stride):
            patch = img[:,y:int(y+win),x:int(x+win)]
            Y[:, :, :, k] = np.array(patch[:])
            location[:,k] = [x, y]
            k = k + 1
    return Y, location      # [1, H, W, P](0-1)float32

def Im2Resize(img, win, stride=1):# [H, W](0-255)int8
    img = cv2.resize(img, (win, win))
    img = img.astype(np.float32)[None, :, :, None]/255.
    return img # [1, H, W, 1](0-1)float32

def img_clip_normalize(imglist):
    for img in imglist:
        batch_size, c, h, w = img.shape
        img = img[:,:,0+1:h-1,0+1:w-1]
    return imglist
