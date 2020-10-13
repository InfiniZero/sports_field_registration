import os
import sys
import cv2
import opts
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import lib.model_initH as initH
import lib.model_refineH as refineH
import lib.model_unet as unet


def optimal_fun(args):
    global model_ps, img_dist
    homography_vector = args
    homographyMatrix = homography_vector.reshape((3, 3))
    model_ps_h = cv2.perspectiveTransform(model_ps, homographyMatrix)
    cost = 0
    for i in range(len(model_ps_h[0])):
        p = model_ps_h[0][i]
        if 0 < p[0] < 1280 and 0 < p[1] < 720:
            cost_p = img_dist[int(p[1]), int(p[0])]
            cost += cost_p
    return cost


def main(opt):
    torch.cuda.set_device(0)
    inference(opt)


def inference(opt):
    # pre-process
    print("load model")

    model_init = initH.SFRNet(opt)
    model_init.cuda()
    checkpoint_init_resume = opt["weights_path"] + "/initH.pth"
    checkpoint_init = torch.load(checkpoint_init_resume)
    model_init.load_state_dict(checkpoint_init['state_dict'])
    model_init.eval()

    model_seg = unet.SFRNet()
    model_seg.cuda()
    model_seg.load_state_dict(checkpoint_init['state_dict'], strict=False)
    model_seg.eval()

    model_refine = refineH.SFRNet(opt)
    model_refine.cuda()
    checkpoint_refine_resume = opt["weights_path"] + "/refineH.pth"
    checkpoint_refine = torch.load(checkpoint_refine_resume)
    model_refine.load_state_dict(checkpoint_refine['state_dict'])
    model_refine.eval()

    test_image = opt['test_image_file']
    field_template_file = opt['field_template_file']
    img_template = cv2.imread(field_template_file)
    img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    img_template_gray_c = cv2.cvtColor(img_template_gray, cv2.COLOR_GRAY2BGR)
    ret, temp_mask = cv2.threshold(
        img_template_gray, 200, 255, cv2.THRESH_BINARY)
    temp_mask_view = img_template.copy()
    for r in range(temp_mask_view.shape[0]):
        for c in range(temp_mask_view.shape[1]):
            if temp_mask[r, c] == 255:
                temp_mask_view[r, c, :] = [255, 255, 0]
            else:
                temp_mask_view[r, c, :] = [0, 0, 0]
    points_list = []
    for r in range(1, temp_mask.shape[0], 7):
        for c in range(1, temp_mask.shape[1], 7):
            if temp_mask[r, c] == 255:
                points_list.append([c, r])
    global model_ps
    model_ps = np.float32([points_list])

    # homography initialization
    img_original = cv2.imread(test_image)
    img_tensor = transforms.ToTensor()(img_original)
    img_tensor = F.interpolate(img_tensor.unsqueeze(
        0), size=256, mode="nearest").squeeze(0)
    img_tensor = img_tensor.unsqueeze(0)

    input_data_seg = img_tensor.cuda()
    with torch.no_grad():
        output_seg = model_seg(input_data_seg)
    del input_data_seg
    torch.cuda.empty_cache()
    output_sigmod = torch.sigmoid(output_seg)
    out_mask = output_sigmod.cpu().numpy()[0]
    out_mask = out_mask.transpose((1, 2, 0))
    out_mask0 = out_mask[:, :, 0]
    out_mask0 = cv2.resize(
        out_mask0, (img_original.shape[1], img_original.shape[0]))

    input_data = img_tensor.cuda()
    with torch.no_grad():
        output = model_init(input_data)
    out_h = output.cpu().numpy()[0]
    del input_data
    torch.cuda.empty_cache()
    height = img_original.shape[0]
    width = img_original.shape[1]
    p_h1 = int(0.5*height)
    p_h2 = int(0.9*height)
    points_l = np.float32(
        [[0, p_h1], [width-1, p_h1], [0, p_h2], [width-1, p_h2]])
    real_h = [0]*8
    real_h[0] = out_h[2] + out_h[0]
    real_h[1] = out_h[3] + out_h[1]
    real_h[2] = out_h[4] + out_h[0]
    real_h[3] = out_h[5] + out_h[1]
    real_h[4] = out_h[6] + out_h[0]
    real_h[5] = out_h[7] + out_h[1]
    real_h[6] = out_h[8] + out_h[0]
    real_h[7] = out_h[9] + out_h[1]
    points_e = np.float32([[real_h[0]*940 + 470, real_h[1]*500 + 250],
                           [real_h[2]*940 + 470, real_h[3]*500 + 250],
                           [real_h[4]*940 + 470, real_h[5]*500 + 250],
                           [real_h[6]*940 + 470, real_h[7]*500 + 250]])
    homographyMatrix_init = cv2.getPerspectiveTransform(points_l, points_e)
    homographyMatrix_init_reproj = cv2.getPerspectiveTransform(
        points_e, points_l)

    # homography refinement
    homographyMatrix_refine = homographyMatrix_init
    homographyMatrix_refine_reproj = homographyMatrix_init_reproj
    out_cost = 0
    for i in range(5):
        img_tran_temp = cv2.warpPerspective(
            temp_mask, homographyMatrix_refine_reproj, (width, height))
        img_tran_temp_resize = cv2.resize(img_tran_temp, (256, 256))
        temp_tensor = transforms.ToTensor()(img_tran_temp_resize)
        input_img_data = img_tensor.cuda()
        input_temp_data = temp_tensor.unsqueeze(0).cuda()
        with torch.no_grad():
            output_refine, output_cost = model_refine(
                input_img_data, input_temp_data)
        out_h_refine = output_refine.cpu().numpy()[0]
        out_h_cost = output_cost.cpu().numpy()[0]
        del input_img_data
        torch.cuda.empty_cache()
        out_cost = out_h_cost[0]
        out_h = out_h - out_h_refine
        real_h = [0]*8
        real_h[0] = out_h[2] + out_h[0]
        real_h[1] = out_h[3] + out_h[1]
        real_h[2] = out_h[4] + out_h[0]
        real_h[3] = out_h[5] + out_h[1]
        real_h[4] = out_h[6] + out_h[0]
        real_h[5] = out_h[7] + out_h[1]
        real_h[6] = out_h[8] + out_h[0]
        real_h[7] = out_h[9] + out_h[1]
        points_e = np.float32([[real_h[0]*940 + 470, real_h[1]*500 + 250],
                               [real_h[2]*940 + 470, real_h[3]*500 + 250],
                               [real_h[4]*940 + 470, real_h[5]*500 + 250],
                               [real_h[6]*940 + 470, real_h[7]*500 + 250]])
        homographyMatrix_refine = cv2.getPerspectiveTransform(
            points_l, points_e)
        homographyMatrix_refine_reproj = np.linalg.inv(homographyMatrix_refine)

    # distance-map optimization
    ret, out_mask0_thred = cv2.threshold(
        out_mask0, 0.2, 255, cv2.THRESH_BINARY_INV)
    out_mask0_thred = out_mask0_thred.astype(np.uint8)
    global img_dist
    img_dist = cv2.distanceTransform(
        out_mask0_thred, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    res = minimize(optimal_fun, homographyMatrix_refine_reproj,
                   method='nelder-mead', options={'adaptive': True})
    homographyMatrix_refine_reproj_new = res.x.reshape((3, 3))
    homographyMatrix_refine_new = np.linalg.inv(
        homographyMatrix_refine_reproj_new)

    img_tran_temp = cv2.warpPerspective(
        temp_mask, homographyMatrix_refine_reproj_new, (width, height))
    img_tran_temp_resize = cv2.resize(img_tran_temp, (256, 256))
    temp_tensor = transforms.ToTensor()(img_tran_temp_resize)
    input_img_data = img_tensor.cuda()
    input_temp_data = temp_tensor.unsqueeze(0).cuda()
    with torch.no_grad():
        output_refine, output_cost = model_refine(
            input_img_data, input_temp_data)
    out_h_cost = output_cost.cpu().numpy()[0]
    out_cost_new = out_h_cost[0]
    if out_cost_new < out_cost:
        homographyMatrix_refine_reproj = homographyMatrix_refine_reproj_new
        homographyMatrix_refine = homographyMatrix_refine_new

    # homographyMatrix_refine = homographyMatrix_init
    # homographyMatrix_refine_reproj = homographyMatrix_init_reproj

    img_tran_refine_final = cv2.warpPerspective(
        img_original, homographyMatrix_refine, (940, 500))
    img_tran_gray_refine_final = cv2.cvtColor(
        img_tran_refine_final, cv2.COLOR_BGR2GRAY)
    ret, mask_refine_final = cv2.threshold(
        img_tran_gray_refine_final, 10, 255, cv2.THRESH_BINARY_INV)
    img_template_gray_c_refine_final = cv2.bitwise_and(
        img_template_gray_c, img_template_gray_c, mask=mask_refine_final)
    img_view_refine_final = cv2.bitwise_or(
        img_tran_refine_final, img_template_gray_c_refine_final)

    temp_reproj_refine_final = cv2.warpPerspective(
        temp_mask, homographyMatrix_refine_reproj, (width, height))
    temp_reproj_refine_final = cv2.bitwise_not(
        temp_reproj_refine_final, temp_reproj_refine_final)
    temp_view_reproj_refine_final = cv2.warpPerspective(
        temp_mask_view, homographyMatrix_refine_reproj, (width, height))
    img_tran_reproj_refine_final = cv2.bitwise_and(
        img_original, img_original, mask=temp_reproj_refine_final)
    img_tran_reproj_refine_final = cv2.bitwise_or(
        temp_view_reproj_refine_final, img_tran_reproj_refine_final)

    cv2.imwrite(opt['field_view_file'], img_view_refine_final)
    cv2.imwrite(opt['broadcast_view_file'], img_tran_reproj_refine_final)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    opt['test_image_file'] = './demo/demo.jpg'
    opt['field_view_file'] = './demo/field_view.jpg'
    opt['broadcast_view_file'] = './demo/broadcast_view.jpg'
    main(opt)
