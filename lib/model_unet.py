import torch
import torch.nn as nn
from lib.unet import UNet


class SFRNet(nn.Module):

    def __init__(self):
        super(SFRNet, self).__init__()
        self.unet = UNet(n_class=2)

    def forward(self, input):
        output = self.unet(input)
        return output
