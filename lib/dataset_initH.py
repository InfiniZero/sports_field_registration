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
        self.H_vector_list = []
        self.img_list = []

        if self.mode == "train":
            self.image_dir = opt["img_train_dir"]
            self.H_dir = opt["H_train_dir"]
        else:
            self.image_dir = opt["img_test_dir"]
            self.H_dir = opt["H_test_dir"]
        img_list = os.listdir(self.image_dir)
        img_list.sort(key=lambda x: (int(x[:2])*1000+int(x[3:7])))
        H_list = os.listdir(self.H_dir)
        H_list.sort(key=lambda x: (int(x[:2])*1000+int(x[3:7])))
        for i in range(len(H_list)):
            H_path = os.path.join(self.H_dir, H_list[i])
            H_vector = np.array(10)
            f = open(H_path, 'r')
            for line in f:
                temp = np.array(line.strip().split(' '))
                temp = temp.astype(float)
                H_vector = temp
            f.close()
            self.H_vector_list.append(H_vector)
        self.img_list = img_list
        print(len(self.img_list))

    def __getitem__(self, index):
        img_data = self._load_img(index)
        H_data = self._load_H(index)
        sample = {'image': img_data, 'H_vector': H_data}
        return sample

    def __len__(self):
        return len(self.img_list)

    def _load_img(self, index):
        img_path = os.path.join(self.image_dir, self.img_list[index])
        img_original = cv2.imread(img_path)
        pil_image = Image.fromarray(img_original)
        if self.mode == "train":
            im_aug = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0),
                transforms.ToTensor()
            ])
        else:
            im_aug = transforms.Compose([
                transforms.ToTensor()
            ])
        img_tensor = im_aug(pil_image)
        img_tensor = F.interpolate(img_tensor.unsqueeze(
            0), size=256, mode="nearest").squeeze(0)
        return img_tensor

    def _load_H(self, index):
        H_vector = self.H_vector_list[index]
        H_data = torch.from_numpy(H_vector).to(torch.float32)
        return H_data
