import torch
import torch.nn as nn
from lib.unet import UNet
from lib.resnet import resnet50


class SFRNet(nn.Module):

    def __init__(self, opt):
        super(SFRNet, self).__init__()
        self.unet = UNet(n_class=2)
        for param in self.unet.parameters():
            param.requires_grad = False
        self.resnet = resnet50(opt, pretrained_all=False, input_dim=2)
        self.fc_box = nn.Sequential(
            nn.Linear(2048, 10)
        )

    def forward(self, input):
        segmap = self.unet(input)
        segmap_sigmod = torch.sigmoid(segmap)
        feature = self.resnet(segmap_sigmod)
        output = self.fc_box(feature)
        return output
