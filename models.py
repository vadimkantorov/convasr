import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2LetterRu(nn.Sequential):
    def __init__(self, num_classes, num_input_features = 64):
        def conv_bn_relu_dropout(kernel_size, num_channels, stride = 1, padding = 0, dropout = 0.2, batch_norm_momentum = 0.1):
            return nn.Sequential(
                nn.Conv1d(num_channels[0], num_channels[1], kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum),
                ReLUDropoutInplace(p = dropout)
            )

        layers = [
            conv_bn_relu_dropout(kernel_size = 13, num_channels = (num_input_features, 768), stride = 2, padding = 6),
            conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
            conv_bn_relu_dropout(kernel_size = 31, num_channels = (768, 2048), stride = 1, padding = 15),
            conv_bn_relu_dropout(kernel_size = 1,  num_channels = (2048, 2048), stride = 1, padding = 0),
            nn.Conv1d(2048, num_classes, kernel_size = 1, stride = 1)
        ]
        super(Wav2LetterRu, self).__init__(*layers)

class Wav2LetterVanilla(nn.Sequential):
    def __init__(self, num_classes, num_input_features = 161):
        def conv_bn_clip(kernel_size, num_channels, stride = 1, dilation = 1, repeat = 1, padding = 0):
            modules = []
            for i in range(repeat):
                modules.append(nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding))
                modules.append(nn.Hardtanh(0, 20, inplace = True))
            return nn.Sequential(*modules)

        layers = [
            conv_bn_clip(kernel_size = 11, num_channels = (num_input_features, 256), stride = 2, padding = 5), # 64
            conv_bn_clip(kernel_size = 11, num_channels = (256, 256), repeat = 3, padding = 5),
            conv_bn_clip(kernel_size = 13, num_channels = (256, 384), repeat = 3, padding = 6),
            conv_bn_clip(kernel_size = 17, num_channels = (384, 512), repeat = 3, padding = 8),
            conv_bn_clip(kernel_size = 21, num_channels = (512, 640), repeat = 3, padding = 10),
            conv_bn_clip(kernel_size = 25, num_channels = (640, 768), repeat = 3, padding = 12),
            conv_bn_clip(kernel_size = 29, num_channels = (768, 896), repeat = 1, padding = 28, dilation = 2),
            conv_bn_clip(kernel_size = 1, num_channels = (896, 1024), repeat = 1),
            nn.Conv1d(1024, num_classes, kernel_size = 1)
        ]

        super(Wav2LetterVanilla, self).__init__(*layers)

class JasperNet(nn.ModuleList):
    def __init__(self, num_classes, num_input_features = 161):
        def conv_bn_relu_dropout_residual(kernel_size, num_channels, dropout = 0, stride = 1, dilation = 1, padding = 0, batch_norm_momentum = 0.1, repeat = 1, num_channels_residual = []):
                return nn.ModuleDict(dict(
                    relu_dropout = ReLUDropoutInplace(p = dropout),
                    conv = nn.ModuleList([nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding, bias = False) for i in range(repeat)]),
                    bn = nn.ModuleList([nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for i in range(repeat)]),
                    conv_residual = nn.ModuleList([nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) for in_channels in num_channels_residual]),
                    bn_residual = nn.ModuleList([nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for in_channels in num_channels_residual])
                ))

        blocks = [
            conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (num_input_features, 256), dropout = 0.2, padding = 5, stride = 2),

            #conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (256, 256), dropout = 0.2, padding = 5, repeat = 5, num_channels_residual = [256]),
            #conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (256, 256), dropout = 0.2, padding = 5, repeat = 5, num_channels_residual = [256, 256]),
            #conv_bn_relu_dropout_residual(kernel_size = 13, num_channels = (256, 384), dropout = 0.2, padding = 6, repeat = 5, num_channels_residual = [256, 256, 256]),
            #conv_bn_relu_dropout_residual(kernel_size = 13, num_channels = (384, 384), dropout = 0.2, padding = 6, repeat = 5, num_channels_residual = [256, 256, 256, 384]),
            #conv_bn_relu_dropout_residual(kernel_size = 17, num_channels = (384, 512), dropout = 0.2, padding = 8, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384]),
            #conv_bn_relu_dropout_residual(kernel_size = 17, num_channels = (512, 512), dropout = 0.2, padding = 8, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512]),
            #conv_bn_relu_dropout_residual(kernel_size = 21, num_channels = (512, 640), dropout = 0.3, padding = 10, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512]),
            #conv_bn_relu_dropout_residual(kernel_size = 21, num_channels = (640, 640), dropout = 0.3, padding = 10, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640]),
            #conv_bn_relu_dropout_residual(kernel_size = 25, num_channels = (640, 768), dropout = 0.3, padding = 12, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640]),
            #conv_bn_relu_dropout_residual(kernel_size = 25, num_channels = (768, 768), dropout = 0.3, padding = 12, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640, 768]),

            conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (256, 256), dropout = 0.2, padding = 5,  repeat = 3, num_channels_residual = [256]),
            conv_bn_relu_dropout_residual(kernel_size = 13, num_channels = (256, 384), dropout = 0.2, padding = 6,  repeat = 3, num_channels_residual = [256, 256]),
            conv_bn_relu_dropout_residual(kernel_size = 17, num_channels = (384, 512), dropout = 0.2, padding = 8,  repeat = 3, num_channels_residual = [256, 256, 384]),
            conv_bn_relu_dropout_residual(kernel_size = 21, num_channels = (512, 640), dropout = 0.3, padding = 10, repeat = 3, num_channels_residual = [256, 256, 384, 512]),
            conv_bn_relu_dropout_residual(kernel_size = 25, num_channels = (640, 768), dropout = 0.3, padding = 12, repeat = 3, num_channels_residual = [256, 256, 384, 512, 640]),
            conv_bn_relu_dropout_residual(kernel_size = 29, num_channels = (768, 896), dropout = 0.4, padding = 28, dilation = 2),
            conv_bn_relu_dropout_residual(kernel_size = 1, num_channels = (896, 1024), dropout = 0.4),

            nn.Conv1d(1024, num_classes, kernel_size = 1)
        ]
        super(JasperNet, self).__init__(blocks)

    def forward(self, x):
        residual = []
        for i, block in enumerate(list(self)[:-1]):
            for conv, bn in zip(block.conv[:-1], block.bn[:-1]):
                x = bn(conv(x))
                x = block.relu_dropout(x)
            x = block.bn[-1](block.conv[-1](x))
            for conv, bn, r in zip(block.conv_residual, block.bn_residual, residual if i < len(self) - 3 else []):
                x = x + bn(conv(r))
            x = block.relu_dropout(x)
            residual.append(x)
        return self[-1](x)

class Speech2TextModel(nn.Module):
    def __init__(self, model):
        super(Speech2TextModel, self).__init__()
        self.model = model

    def forward(self, x, lengths):
        output_lengths = lengths.int() // 2
        logits = self.model(x)
        return logits, output_lengths

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

def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location = 'cpu')
    model.load_state_dict(state_dict)

def save_checkpoint(model, checkpoint_path):
    state_dict = model.state_dict()
    torch.save(state_dict, checkpoint_path)
