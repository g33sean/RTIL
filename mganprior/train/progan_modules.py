import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt
import numpy as np


class inter_dist(nn.Module):
    '''
        Sample intermediate layer distribution based on mganprior
    '''
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
    def __init__(self,args,input_code_dim, in_channel, pixel_norm):
        super().__init__()
        self.args=args
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

        self.inter_sample=inter_dist()
        
        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1-alpha)*skip_rgb + alpha*module2(feat2)
        else:
            out = module2(feat2)
        return out

    def forward(self,z,z1,z2, step=0, alpha=-1,rtil=None):
        if step > self.max_step:
            step = self.max_step

        if rtil==0:

        
            out_4 = self.input_layer(z .view(-1, self.input_dim, 1, 1))
            out_4 = self.progression_4(out_4)

           

            if step ==0:
                
                return self.to_rgb_4(out_4)

            out_8 = self.progress(out_4, self.progression_8)

            if step==1:
                return self.output( out_4, out_8, self.to_rgb_4, self.to_rgb_8, alpha )
            
            out_16 = self.progress(out_8, self.progression_16)
            
            if step==2:
               
                return self.output( out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha )
            
            out_32 = self.progress(out_16, self.progression_32)
            if step==3:
                return self.output( out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha )

            out_64 = self.progress(out_32, self.progression_64)
            if step==4:
                return self.output( out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha )
            
            out_128 = self.progress(out_64, self.progression_128)
            if step==5:
                return self.output( out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha )

            out_256 = self.progress(out_128, self.progression_256)
            if step==6:
                return self.output( out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha )
        
        if rtil==1:
            z1=torch.chunk(z1,self.args.z_number[0])
            feat_list=[]

            for z1 in z1:
                out_4 = self.input_layer(z1.view(-1, self.input_dim, 1, 1))
                out_4 = self.progression_4(out_4)
                if step ==0:
                    feat_list.append(out_4)
                if step >0:
                    out_8 = self.progress(out_4, self.progression_8)
                if step>1:
                    out_16=self.progress(out_8,self.progression_16)
                if step>2:
                    out_32=self.progress(out_16,self.progression_32)

                if step ==1:
                    feat_list.append(out_8)

                if step ==2:
                    feat_list.append(out_16)

                if step >=3:
                     feat_list.append(out_32)

            if step ==0:
                out_4=self.inter_sample(feat_list,self.args.z_number[0],out_4)
                feat_list.clear()
                return self.to_rgb_4(out_4)
   
            if step==1:
                out_8=self.inter_sample(feat_list,self.args.z_number[0],out_8)
                feat_list.clear()
                return self.output(out_4,out_8, self.to_rgb_4, self.to_rgb_8, alpha )

            if step ==2:
                out_16=self.inter_sample(feat_list,self.args.z_number[0],out_16)
                feat_list.clear()
                return self.output(out_8,out_16, self.to_rgb_8, self.to_rgb_16, alpha )
            
            if step==3:
                out_32=self.inter_sample(feat_list,self.args.z_number[0],out_32)
                feat_list.clear()
                return self.output(out_16,out_32, self.to_rgb_16, self.to_rgb_32, alpha )

            out_32=self.inter_sample(feat_list,self.args.z_number[0],out_32)
            feat_list.clear()
            out_64 = self.progress(out_32, self.progression_64)
            if step==4:
                return self.output( out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha )
            
            out_128 = self.progress(out_64, self.progression_128)
            if step==5:
                return self.output( out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha )

            out_256 = self.progress(out_128, self.progression_256)
            if step==6:
                return self.output( out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha )

        if rtil==2:
            z2=torch.chunk(z2,self.args.z_number[1])
            feat_list=[]

            for z2 in z2:
                out_4 = self.input_layer(z2.view(-1, self.input_dim, 1, 1))
                out_4 = self.progression_4(out_4)
                if step ==0:
                    feat_list.append(out_4)
                if step >0:
                    out_8 = self.progress(out_4, self.progression_8)
                if step>1:
                    out_16=self.progress(out_8,self.progression_16)
                if step>2:
                    out_32=self.progress(out_16,self.progression_32)

                if step ==1:
                    feat_list.append(out_8)

                if step ==2:
                    feat_list.append(out_16)

                if step >=3:
                     feat_list.append(out_32)

            if step ==0:
                out_4=self.inter_sample(feat_list,self.args.z_number[1],out_4)
                feat_list.clear()
                return self.to_rgb_4(out_4)
   
            if step==1:
                out_8=self.inter_sample(feat_list,self.args.z_number[1],out_8)
                feat_list.clear()
                return self.output(out_4,out_8, self.to_rgb_4, self.to_rgb_8, alpha )

            if step ==2:
                out_16=self.inter_sample(feat_list,self.args.z_number[1],out_16)
                feat_list.clear()
                return self.output(out_8,out_16, self.to_rgb_8, self.to_rgb_16, alpha )
            
            if step==3:
                out_32=self.inter_sample(feat_list,self.args.z_number[1],out_32)
                feat_list.clear()
                return self.output(out_16,out_32, self.to_rgb_16, self.to_rgb_32, alpha )

            out_32=self.inter_sample(feat_list,self.args.z_number[1],out_32)
            feat_list.clear()
            out_64 = self.progress(out_32, self.progression_64)
            if step==4:
                return self.output( out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha )
            
            out_128 = self.progress(out_64, self.progression_128)
            if step==5:
                return self.output( out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha )

            out_256 = self.progress(out_128, self.progression_256)
            if step==6:
                return self.output( out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha )

                
            




class Discriminator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(feat_dim//4, feat_dim//4, 3, 1),
                                          ConvBlock(feat_dim//4, feat_dim//2, 3, 1),
                                          ConvBlock(feat_dim//2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim+1, feat_dim, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, feat_dim//4, 1),
                                       EqualConv2d(3, feat_dim//4, 1),
                                       EqualConv2d(3, feat_dim//2, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out
