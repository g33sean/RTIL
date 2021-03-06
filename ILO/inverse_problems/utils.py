import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import os

def partial_circulant_torch(inputs, filters, indices, sign_pattern):
    n = np.prod(inputs.shape[1:])
    bs = inputs.shape[0]
    input_reshape = inputs.reshape(bs, n)
    input_sign = input_reshape * sign_pattern
    def to_complex(tensor):
        zeros = torch.zeros_like(tensor)
        concat = torch.cat((tensor, zeros), axis=0)
        reshape = concat.view(2, -1, n)
        return reshape.permute(1, 2, 0)
    complex_input = to_complex(input_sign)
    complex_filter = to_complex(filters)
    input_fft = torch.fft(complex_input, 1)
    filter_fft = torch.fft(complex_filter, 1)
    output_fft = torch.zeros_like(input_fft)
    output_fft[:,:,0] = input_fft[:,:,0]*filter_fft[:,:,0] - input_fft[:,:,1] * filter_fft[:,:,1]
    output_fft[:,:,1] = input_fft[:,:,1] * filter_fft[:,:,0] + input_fft[:,:,0] * filter_fft[:,:,1]
    output_ifft = torch.ifft(output_fft, 1)
    output_real = output_ifft[:,:,0]
    return output_real[:, indices]




def random_inp_torch(image,mask):
    return image*mask


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0

    def __init__(self, factor=4, cuda=True, padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                          for i in range(size)], dtype=torch.float32)
        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.cuda = '.cuda' if cuda else ''
        self.padding = padding
        #self.padding = 'constant'
        #self.padding = 'replicate'
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor
        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1.type('torch{}.FloatTensor'.format(self.cuda))
        filters2 = self.k2.type('torch{}.FloatTensor'.format(self.cuda))
        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW
        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x.float(), weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x




