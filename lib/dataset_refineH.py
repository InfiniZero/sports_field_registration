import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import random


class ImageDataset(Dataset):

    def __init__(self, opt):
        self.mode = opt["mode"]
        self.H_vector_list = []
        self.img_list = []
        self.field_template_file = opt["field_template_file"]
        img_template = cv2.imread(self.field_template_file)
        img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        ret, temp_mask = cv2.threshold(
            img_template_gray, 200, 255, cv2.THRESH_BINARY)
        ret, temp_mask_all = cv2.threshold(
            img_template_gray, 5, 255, cv2.THRESH_BINARY)
        self.temp_mask = temp_mask
        self.temp_mask_all = temp_mask_all

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
        img_data, img_temp_data, deltaH_data, iou_data = self._load_data(index)
        sample = {'image': img_data, 'temp': img_temp_data,
                  'deltaH': deltaH_data, 'iou': iou_data}
        return sample

    def __len__(self):
        return len(self.img_list)

    def _load_data(self, index):
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

        H_vector = self.H_vector_list[index]
        height = img_original.shape[0]
        width = img_original.shape[1]
        p_h1 = int(0.5*height)
        p_h2 = int(0.9*height)
        points_l = np.float32(
            [[0, p_h1], [width-1, p_h1], [0, p_h2], [width-1, p_h2]])
        real_h = [0]*8
        real_h[0] = H_vector[2] + H_vector[0]
        real_h[1] = H_vector[3] + H_vector[1]
        real_h[2] = H_vector[4] + H_vector[0]
        real_h[3] = H_vector[5] + H_vector[1]
        real_h[4] = H_vector[6] + H_vector[0]
        real_h[5] = H_vector[7] + H_vector[1]
        real_h[6] = H_vector[8] + H_vector[0]
        real_h[7] = H_vector[9] + H_vector[1]
        points_e_gt = np.float32([[real_h[0]*940 + 470, real_h[1]*500 + 250],
                                  [real_h[2]*940 + 470, real_h[3]*500 + 250],
                                  [real_h[4]*940 + 470, real_h[5]*500 + 250],
                                  [real_h[6]*940 + 470, real_h[7]*500 + 250]])
        homographyMatrix_gt = cv2.getPerspectiveTransform(
            points_l, points_e_gt)
        homographyMatrix_gt_reproj = np.linalg.inv(homographyMatrix_gt)
        img_tran_temp_all_gt_reproj = cv2.warpPerspective(
            self.temp_mask_all, homographyMatrix_gt_reproj, (width, height))
        delta_gx = random.uniform(-0.08, 0.08)
        delta_gy = random.uniform(-0.08, 0.08)
        delta_l1x = random.uniform(-0.04, 0.04)
        delta_l1y = random.uniform(-0.04, 0.04)
        delta_l2x = random.uniform(-0.04, 0.04)
        delta_l2y = random.uniform(-0.04, 0.04)
        delta_l3x = random.uniform(-0.04, 0.04)
        delta_l3y = random.uniform(-0.04, 0.04)
        delta_l4x = random.uniform(-0.04, 0.04)
        delta_l4y = random.uniform(-0.04, 0.04)
        deltaH_vector = np.array([delta_gx, delta_gy, delta_l1x, delta_l1y,
                                  delta_l2x, delta_l2y, delta_l3x, delta_l3y,
                                  delta_l4x, delta_l4y])
        deltaH_data = torch.from_numpy(deltaH_vector).to(torch.float32)
        H_vector_random = real_h.copy()
        H_vector_random[0] = H_vector_random[0] + delta_gx + delta_l1x
        H_vector_random[1] = H_vector_random[1] + delta_gy + delta_l1y
        H_vector_random[2] = H_vector_random[2] + delta_gx + delta_l2x
        H_vector_random[3] = H_vector_random[3] + delta_gy + delta_l2y
        H_vector_random[4] = H_vector_random[4] + delta_gx + delta_l3x
        H_vector_random[5] = H_vector_random[5] + delta_gy + delta_l3y
        H_vector_random[6] = H_vector_random[6] + delta_gx + delta_l4x
        H_vector_random[7] = H_vector_random[7] + delta_gy + delta_l4y

        points_e = np.float32([[H_vector_random[0]*940 + 470, H_vector_random[1]*500 + 250],
                               [H_vector_random[2]*940 + 470,
                                   H_vector_random[3]*500 + 250],
                               [H_vector_random[4]*940 + 470,
                                   H_vector_random[5]*500 + 250],
                               [H_vector_random[6]*940 + 470, H_vector_random[7]*500 + 250]])
        homographyMatrix = cv2.getPerspectiveTransform(points_l, points_e)
        homographyMatrix_reproj = np.linalg.inv(homographyMatrix)
        img_tran_temp = cv2.warpPerspective(
            self.temp_mask, homographyMatrix_reproj, (width, height))
        img_tran_temp_resize = cv2.resize(img_tran_temp, (256, 256))
        img_temp_data = transforms.ToTensor()(img_tran_temp_resize)

        img_tran_temp_all_reproj = cv2.warpPerspective(
            self.temp_mask_all, homographyMatrix_reproj, (width, height))
        img_tran_iou_reproj_and = cv2.bitwise_and(
            img_tran_temp_all_reproj, img_tran_temp_all_gt_reproj)
        img_tran_iou_reproj_or = cv2.bitwise_or(
            img_tran_temp_all_reproj, img_tran_temp_all_gt_reproj)
        iou_and = cv2.countNonZero(img_tran_iou_reproj_and)
        iou_or = cv2.countNonZero(img_tran_iou_reproj_or)
        iou_error = 1 - iou_and/iou_or
        iou_data = torch.from_numpy(np.array([iou_error])).to(torch.float32)

        return img_tensor, img_temp_data, deltaH_data, iou_data
