import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as kgt
from einops import rearrange
import numbers
import numpy as np
from unet import UNet

class Codec(nn.Module):
    def __init__(self, num_iter=4, channel=1, num_filters=64):
        super(Codec, self).__init__()
        self.CSB = LCSC(down_ch=channel, up_ch=num_filters, num_iter=num_iter)
    def forward(self, x, mode ='encode'):# mode['encode', 'train','decode']
        if mode == 'decode':
            return self.CSB(x, mode='decode')
        elif mode == 'encode':
            return self.CSB(x, mode='test')
        else:
            return self.CSB(x, mode='train')

class Filter(nn.Module):
    def __init__(self, num_filters=64, num_heads=8, inn_iter=4):
        super(Filter, self).__init__()
        self.SF = HFF(dim=num_filters, num_heads = num_heads)
        self.DF = IFF(channels=num_filters, num_layers=inn_iter)
    def forward(self, x):
        sfx, dfx = self.SF(x), self.DF(x)
        return sfx, dfx
    
class Fusion(nn.Module):
    def __init__(self, num_filters=64):
        super(Fusion, self).__init__()
        self.GD = GD(in_features=num_filters*4, ffn_expansion_factor=1)
        self.fuser = nn.Conv2d(in_channels=num_filters*4, out_channels=num_filters,
                               kernel_size=3, padding=1, stride=1)
        nn.init.xavier_uniform_(self.fuser.weight.data)
    def forward(self, x):
        ff = self.fuser(self.GD(x))
        return ff

class Registration(nn.Module):
    def __init__(self, num_filters=64, patch_size=128):
        super(Registration, self).__init__()
        self.RM = RM(in_channels=num_filters, batch_norm=True, patch_size=patch_size)
    def forward(self, x, y, size):
        return self.RM(x, y, size)
            
            

class CSCDecoder(nn.Module):
    def __init__(self, down_ch=1, up_ch=64, kernel_size=3, padding=1, stride=1):
        super(CSCDecoder, self).__init__()
        # 64->1
        layer_da = Attention(dim=up_ch, num_heads=8, D_kernel_size=3, bias=False)
        layer_dc = nn.Conv2d(in_channels=up_ch, out_channels=down_ch,
                            kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        nn.init.xavier_uniform_(layer_dc.weight.data)
        self.layer_d = nn.Sequential(layer_da, layer_dc)
    def forward(self, x):
        return self.layer_d(x)

class CSCEncoder(nn.Module):
    def __init__(self, down_ch=1, up_ch=64, kernel_size=3, padding=1, stride=1):
        super(CSCEncoder, self).__init__()
        # 1->64
        layer_uc = nn.Conv2d(in_channels=down_ch, out_channels=up_ch,
                            kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        nn.init.xavier_uniform_(layer_uc.weight.data)
        layer_ua = Attention(dim=up_ch, num_heads=8, D_kernel_size=3, bias=False)
        self.layer_u = nn.Sequential(layer_uc, layer_ua)
    def forward(self, x):
        return self.layer_u(x)


class LCSC(nn.Module):
    def __init__(self, down_ch=1, up_ch=64, kernel_size=3, padding=1, stride=1, num_iter=4):
        super(LCSC, self).__init__()
        self.layer_d = CSCDecoder(down_ch=down_ch, up_ch=up_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.layer_u = CSCEncoder(down_ch=down_ch, up_ch=up_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.lam = nn.Parameter(torch.Tensor([0.01]))
        self.num_iter = num_iter
    def forward(self, ts_in, mode ='train'):#mode:[train, test, decode]
        if mode == 'decode':
            return self.layer_d(ts_in)
        else:
            ref = self.layer_u(ts_in)
            ref = torch.mul(torch.sign(ref), F.relu(torch.abs(ref) - self.lam))
            if mode == 'test':
                return ref
            fc = ref
            for _ in range(self.num_iter):
                fc = torch.add(fc - self.layer_u(self.layer_d(fc)), ref)
                fc = torch.mul(torch.sign(fc), F.relu(torch.abs(fc) - self.lam))
            return fc

class HFF(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(HFF, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim, num_heads=num_heads, bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = GD(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class IFF(nn.Module):
    def __init__(self, channels=64, num_layers=3):
        super(IFF, self).__init__()
        INNmodules = [DetailNode(channels=channels) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

def delta_to_H(image_size, delta):
    corners = torch.tensor(
            [
                [0,0],
                [0 + image_size, 0],
                [0 + image_size, 0 + image_size],
                [0, 0 + image_size],
            ], dtype=torch.float)
    corners_hat = corners + delta
    h = kgt.get_perspective_transform(corners, corners_hat)
    return h

class RM(nn.Module):
    def __init__(self, in_channels=64, batch_norm=False,patch_size=128):
        super(RM, self).__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.unet = UNet(in_channels, 1)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(4 * (patch_size//(in_channels//8)) * (patch_size//(in_channels//8)), 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 4 * 2),
        )

    def forward(self, x, y, size):#[B,1,H,W],[B,1,H,W],[B,2]
        H, W = size
        max_size = max(H, W)
        x1 = x[:,:,H//2-max_size//2:H//2+max_size//2,W//2-max_size//2:W//2+max_size//2]
        y1 = y[:,:,H//2-max_size//2:H//2+max_size//2,W//2-max_size//2:W//2+max_size//2]
        xi = F.interpolate(x1, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=True)
        yi = F.interpolate(y1, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=True)
        xi = self.conv1(x)
        yi = self.conv2(y)
        end, center = self.unet(torch.cat((xi, yi), dim=1))
        field = end
        delta = self.fc(center)
        delta = delta.view(-1,4,2)
        batch_size = delta.size(0)
        corners = torch.tensor(
            [
                [0,0],
                [0 + self.patch_size, 0],
                [0 + self.patch_size, 0 + self.patch_size],
                [0, 0 + self.patch_size],
            ], dtype = torch.float
        )
        corners = corners.unsqueeze(0).expand(batch_size, 4, 2).float().cuda()
        corners_hat = corners + delta * self.patch_size
        h = kgt.get_perspective_transform(corners, corners_hat)
        h = torch.inverse(h)
        return field, h

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class CNNBlock(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(CNNBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class GD(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 1,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        dim = torch.Size(dim)
        assert len(dim) == 1
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.dim = dim
        self.LNType = LayerNorm_type
    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.LNType == 'BiasFree':
            x = x / torch.sqrt(sigma+1e-5) * self.weight
        else:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self, channels=64):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=channels//2, oup=channels//2, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=channels//2, oup=channels//2, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=channels//2, oup=channels//2, expand_ratio=2)
        self.shffleconv = nn.Conv2d(channels, channels, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, D_kernel_size=3, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape    # [8,64,64,64]
        qkv = self.qkv1(x)      # [8,192,64,64]
        qkv = self.qkv2(qkv)    # [8,192,64,64]
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out

if __name__ == '__main__':
    Codec()
    