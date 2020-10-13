import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, opt):
        self.mode = opt["mode"]
        self.mask_list = []
        self.img_list = []

        if self.mode == "train":
            self.image_dir = opt["img_train_dir"]
            self.mask_dir_0 = opt["mask_train_dir"] + '/0'
            self.mask_dir_1 = opt["mask_train_dir"] + '/1'
        else:
            self.image_dir = opt["img_test_dir"]
            self.mask_dir_0 = opt["mask_test_dir"] + '/0'
            self.mask_dir_1 = opt["mask_test_dir"] + '/1'
        img_list = os.listdir(self.image_dir)
        img_list.sort(key=lambda x: (int(x[:2])*1000+int(x[3:7])))
        mask_0_list = os.listdir(self.mask_dir_0)
        mask_0_list.sort(key=lambda x: (int(x[:2])*1000+int(x[3:7])))
        mask_1_list = os.listdir(self.mask_dir_1)
        mask_1_list.sort(key=lambda x: (int(x[:2])*1000+int(x[3:7])))
        for i in range(len(mask_0_list)):
            mask_0_path = os.path.join(self.mask_dir_0, mask_0_list[i])
            mask_1_path = os.path.join(self.mask_dir_1, mask_1_list[i])
            self.mask_list.append([mask_0_path, mask_1_path])
        self.img_list = img_list
        print(len(self.img_list))

    def __getitem__(self, index):
        img_data = self._load_img(index)
        mask_data = self._load_mask(index)
        sample = {'image': img_data, 'mask_data': mask_data}
        return sample

    def __len__(self):
        return len(self.img_list)

    def _load_img(self, index):
        img_path = os.path.join(self.image_dir, self.img_list[index])
        img_original = cv2.imread(img_path)
        pil_image = Image.fromarray(img_original)
        im_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0),
            transforms.ToTensor()
        ])
        img_tensor = im_aug(pil_image)
        img_tensor = F.interpolate(img_tensor.unsqueeze(
            0), size=256, mode="nearest").squeeze(0)
        return img_tensor

    def _load_mask(self, index):
        mask_0_path = self.mask_list[index][0]
        mask_1_path = self.mask_list[index][1]
        mask_0 = cv2.imread(mask_0_path, cv2.IMREAD_GRAYSCALE)
        mask_1 = cv2.imread(mask_1_path, cv2.IMREAD_GRAYSCALE)
        cv2.normalize(mask_0, mask_0, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(mask_1, mask_1, 0, 1, cv2.NORM_MINMAX)
        mask_0_data = torch.from_numpy(mask_0).to(torch.float32).unsqueeze(0)
        mask_1_data = torch.from_numpy(mask_1).to(torch.float32).unsqueeze(0)
        mask_data = torch.cat([mask_0_data, mask_1_data])
        mask_data = F.interpolate(mask_data.unsqueeze(
            0), size=256, mode="nearest").squeeze(0)
        return mask_data
