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
        self.resnet = resnet50(opt, pretrained_all=True)
        self.fc_box_1 = nn.Sequential(
            nn.Linear(2048, 10)
        )
        self.fc_box_2 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        segmap = self.unet(input1)
        segmap_sigmod = torch.sigmoid(segmap)
        imgmap = torch.cat([segmap_sigmod, input2], dim=1)
        feature = self.resnet(imgmap)
        output = self.fc_box_1(feature)
        iou = self.fc_box_2(feature)
        return output, iou
