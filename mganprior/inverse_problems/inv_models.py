import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt

class G_multi(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,feat_list,z_number,layer):
        device=layer.device
        feat_layer=torch.cat(feat_list,dim=0).reshape(layer.size(0),z_number,layer.size(1),layer.size(2),layer.size(2))
        dist=torch.from_numpy(np.random.dirichlet(np.ones(z_number),size=layer.size(0))).type(dtype=torch.FloatTensor).to(device) 
        x=(feat_layer* dist[:,:,None,None,None]).sum(dim=1)
        
        return x


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None, pixel_norm=True):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        convs = [EqualConv2d(in_channel, out_channel, kernel1, padding=pad1)]
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.2))
        convs.append(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.2))

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)

class Generator(nn.Module):
    def __init__(self, input_code_dim, in_channel, pixel_norm):
        super().__init__()
        self.input_dim = input_code_dim
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(in_channel, in_channel//2, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(in_channel//2, in_channel//4, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(in_channel//4, in_channel//4, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_4 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_64 = EqualConv2d(in_channel//2, 3, 1)
        self.to_rgb_128 = EqualConv2d(in_channel//4, 3, 1)
        self.to_rgb_256 = EqualConv2d(in_channel//4, 3, 1)


    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        out = module(out)
        return out


    def forward(self,z,alpha,z_number,config):
        #z=torch.chunk(z,z_number)
        feat_list=[]
        if config['layer_in']==2:
            for z in z:
                out_4 = self.input_layer(z.view(-1, self.input_dim, 1, 1))
                out_4 = self.progression_4(out_4)
                out_8 = self.progress(out_4, self.progression_8)
                feat_list.append(out_8)
            feat_layer=torch.cat(feat_list,dim=0).reshape(out_8.size(0),z_number,out_8.size(1),out_8.size(2),out_8.size(2))
            x=(feat_layer* alpha[:,:,:,None,None]).sum(dim=1)
            feat_list.clear()
            out_16 = self.progress(x,self.progression_16)
            out_32 = self.progress(out_16, self.progression_32)
            out_64 = self.progress(out_32, self.progression_64)
            out_128 = self.progress(out_64, self.progression_128)
            out_256 = self.progress(out_128, self.progression_256)
            out=self.to_rgb_256(out_256)

        elif config['layer_in']==3:
            for z in z:
                out_4 = self.input_layer(z.view(-1, self.input_dim, 1, 1))
                out_4 = self.progression_4(out_4)
                out_8 = self.progress(out_4, self.progression_8)
                out_16 = self.progress(out_8,self.progression_16)
                feat_list.append(out_16)
            feat_layer=torch.cat(feat_list,dim=0).reshape(out_16.size(0),z_number,out_16.size(1),out_16.size(2),out_16.size(2))
            x=(feat_layer* alpha[:,:,:,None,None]).sum(dim=1)
            feat_list.clear()
            out_32 = self.progress(x, self.progression_32)
            out_64 = self.progress(out_32, self.progression_64)
            out_128 = self.progress(out_64, self.progression_128)
            out_256 = self.progress(out_128, self.progression_256)
            out=self.to_rgb_256(out_256)


        elif config['layer_in']==4:
            for z in z:
                out_4 = self.input_layer(z.view(-1, self.input_dim, 1, 1))
                out_4 = self.progression_4(out_4)
                out_8 = self.progress(out_4, self.progression_8)
                out_16 = self.progress(out_8, self.progression_16)
                out_32 = self.progress(out_16, self.progression_32)
                feat_list.append(out_32)
            feat_layer=torch.cat(feat_list,dim=0).reshape(out_32.size(0),z_number,out_32.size(1),out_32.size(2),out_32.size(2))
            x=(feat_layer* alpha[:,:,:,None,None]).sum(dim=1)
            feat_list.clear()
            out_64 = self.progress(x, self.progression_64)
            out_128 = self.progress(out_64, self.progression_128)
            out_256 = self.progress(out_128, self.progression_256)
            out=self.to_rgb_256(out_256)

        elif config['layer_in']==5:
            for z in z:
                out_4 = self.input_layer(z.view(-1, self.input_dim, 1, 1))
                out_4 = self.progression_4(out_4)
                out_8 = self.progress(out_4, self.progression_8)
                out_16 = self.progress(out_8, self.progression_16)
                out_32 = self.progress(out_16, self.progression_32)
                out_64=self.progress(out_32, self.progression_64)
                feat_list.append(out_64)
            feat_layer=torch.cat(feat_list,dim=0).reshape(out_64.size(0),z_number,out_64.size(1),out_64.size(2),out_64.size(2))
            x=(feat_layer* alpha[:,:,:,None,None]).sum(dim=1)
            feat_list.clear()
            out_128 = self.progress(x, self.progression_128)
            out_256 = self.progress(out_128, self.progression_256)
            out=self.to_rgb_256(out_256)

        elif config['layer_in']==6:
            for z in z:
                out_4 = self.input_layer(z.view(-1, self.input_dim, 1, 1))
                out_4 = self.progression_4(out_4)
                out_8 = self.progress(out_4, self.progression_8)
                out_16 = self.progress(out_8, self.progression_16)
                out_32 = self.progress(out_16, self.progression_32)
                out_64=self.progress(out_32, self.progression_64)
                out_128 = self.progress(out_64, self.progression_128)
                feat_list.append(out_128)
            feat_layer=torch.cat(feat_list,dim=0).reshape(out_128.size(0),z_number,out_128.size(1),out_128.size(2),out_128.size(2))
            x=(feat_layer* alpha[:,:,:,None,None]).sum(dim=1)
            feat_list.clear()
            out_256 = self.progress(x, self.progression_256)
            out=self.to_rgb_256(out_256)

        else:
            print('input layer not defined')

        return (out+1)/2


