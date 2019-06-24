import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2LetterVanilla(nn.Sequential):
    def __init__(self, num_classes):
        def conv_block(kernel_size, num_channels, stride = 1, dilation = 1, repeat = 1, padding = 0):
            modules = []
            for i in range(repeat):
                conv = Conv1dSamePadding(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation)#, padding = padding)
                modules.append(conv)
                modules.append(nn.Hardtanh(0, 20, inplace = True))
            return nn.Sequential(*modules)

        layers = [
            conv_block(kernel_size = 11, num_channels = (161, 256), stride = 2, padding = 5), # 64
            conv_block(kernel_size = 11, num_channels = (256, 256), repeat = 3, padding = 5),
            conv_block(kernel_size = 13, num_channels = (256, 384), repeat = 3, padding = 6),
            conv_block(kernel_size = 17, num_channels = (384, 512), repeat = 3, padding = 8),
            conv_block(kernel_size = 21, num_channels = (512, 640), repeat = 3, padding = 10),
            conv_block(kernel_size = 25, num_channels = (640, 768), repeat = 3, padding = 12),
            conv_block(kernel_size = 29, num_channels = (768, 896), repeat = 1, padding = 28, dilation = 2),
            conv_block(kernel_size = 1, num_channels = (896, 1024), repeat = 1),
            nn.Conv1d(1024, num_classes, 1)
        ]

        super(Wav2LetterVanilla, self).__init__(*layers)

class Speech2TextModel(nn.Module):
    def __init__(self, model):
        super(Speech2TextModel, self).__init__()
        self.model = model

    def forward(self, x, lengths):
        output_lengths = lengths.cpu().int() // 2

        x = x.squeeze(1)
        x = self.model(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        x = x.transpose(0, 1)
        outs = F.softmax(x, dim=-1)
        return x, outs, output_lengths
 
class Conv1dSamePadding(nn.Conv1d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride[0] # just a scalar

    def forward(self, x):
        iw = int(x.size()[-1])
        kw = int(self.weight.size()[-1])
        sw = self.stride
        ow = math.ceil(iw / sw)
        pad_w = max((ow - 1) * self.stride + (kw - 1) * self.dilation[0] + 1 - iw, 0)
        if pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2])
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
