import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUDropoutInplace(torch.nn.Module):
    def __init__(self, p):
        super(ReLUDropoutInplace, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            p1m = 1. - self.p
            mask = torch.rand_like(input) < p1m
            mask *= (input > 0)
            return input.masked_fill_(~mask, 0).mul_(1.0 / p1m)
        else:
            return input.clamp_(min = 0)

class Wav2LetterRu(nn.Sequential):
    def __init__(self, num_classes, dropout = 0.0, batch_norm_momentum = 0.1):
        def conv_block(kernel_size, num_channels, stride = 1, padding = 0):
            return nn.Sequential(
                nn.Conv1d(num_channels[0], num_channels[1], kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum),
                ReLUDropoutInplace(p = dropout)
            )

        layers = [
            conv_block(kernel_size = 13, num_channels = (161, 768), stride = 2, padding = 6),
            conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_block(kernel_size = 31, num_channels = (768, 2048), stride = 1, padding = 15),
            conv_block(kernel_size = 1,  num_channels = (2048, 2048), stride = 1, padding = 0),
            nn.Conv1d(2048, num_classes, kernel_size = 1, stride = 1)
        ]
        super(Wav2LetterRu, self).__init__(nn.Sequential(*[l for s in layers[:-1] for l in s]), layers[-1])

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
        output_lengths = lengths.int() // 2
        logits = self.model(x.squeeze(1))
        logits = logits.permute(2, 0, 1).contiguous().transpose(0, 1)

        return logits, F.softmax(logits, dim=-1), output_lengths

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, checkpoint_dir):
        state_dict = self.model.state_dict()
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        torch.save(state_dict, checkpoint_path)
 
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
