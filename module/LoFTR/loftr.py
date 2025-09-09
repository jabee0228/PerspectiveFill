import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from .util import *
from .findJpg import combine_filenames
from .src.utils.plotting import make_matching_figure
from .src.loftr import LoFTR, default_cfg
import os
import random

def loftrGenerate(img0_pth, img1_pth, blocksXY):
    image_pair = [img0_pth, img1_pth]
    matcher = LoFTR(config=default_cfg)

    image_type = 'outdoor'
    if image_type == 'indoor':
      matcher.load_state_dict(torch.load("/weights/indoor_ds.ckpt")['state_dict'])
    elif image_type == 'outdoor':
      matcher.load_state_dict(torch.load("./module/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    else:
      raise ValueError("Wrong image_type is given.")


    #matcher = matcher.eval().cuda()
    matcher = matcher.eval()
    img0_origin = cv2.imread(image_pair[0])
    img1_origin = cv2.imread(image_pair[1])
    img0_origin = cv2.resize(img0_origin, (640, 480))
    img1_origin = cv2.resize(img1_origin, (640, 480))

    '''
        tmp_pth = "/storage/LoFTR/results/originSet/"+dirName
    if not os.path.exists(tmp_pth+returnBaseName(img0_pth)):
        os.makedirs(tmp_pth, exist_ok=True)
        cv2.imwrite(tmp_pth+"/"+returnBaseName(img0_pth), img0_origin) # store resized origin_img0

    '''


    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None] / 255.
    img1 = torch.from_numpy(img1_raw)[None][None] / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # blocksXY = block_region(mkpts0)
    # blocksXY = generate_random_rectangle(640, 480)
    if (blocksXY == None):
        return
    
    newName = combine_filenames(returnBaseName(img0_pth), returnBaseName(img1_pth))
    # addWhiteBlock(bloackregion[0],bloackregion[1],bloackregion[2],bloackregion[3],"image0","whiteBlock")


    H, status = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)

    result_img0 = img0_origin.copy()
    target_image1 = img1_origin
    result_img0 = cv2.rectangle(result_img0, (blocksXY[0], blocksXY[1]), (blocksXY[2], blocksXY[3]), (255, 255, 255), -1)
    for dst_y in range(blocksXY[1], blocksXY[3]):
        for dst_x in range(blocksXY[0], blocksXY[2]):
            coordinate = np.array([[dst_x], [dst_y], [1]])
            trans_coordinate = np.dot(H, coordinate)
            # 齊次座標轉二維
            trans_coordinate = trans_coordinate / trans_coordinate[2]
            src_x, src_y = int(trans_coordinate[0]), int(trans_coordinate[1])
            if ((src_x > 639 or src_x < 0) or (src_y > 479 or src_y < 0)):
                continue
            elif ((dst_x > 639 or dst_x < 0) or (dst_y > 479 or dst_y < 0)):
                continue
            result_img0[dst_y, dst_x] = target_image1[src_y, src_x] # write pixel

    if not os.path.exists("images060501/origin/"):
        os.makedirs("images060501/origin/", exist_ok=True)
    
    cv2.imwrite("images060501/origin/"+newName, img0_origin)

    if not os.path.exists("images060501/fixed/"):
        os.makedirs("images060501/fixed/", exist_ok=True)
    filename = "images060501/fixed/"+ newName
        # Save the image using OpenCV's imwrite function
    cv2.imwrite(filename, result_img0)
    filename = "images060501/masks/"
    if not os.path.exists("images060501/masks/"):
        os.makedirs("images060501/masks/", exist_ok=True)
    filename += newName
    makeBlackImg(blocksXY[0],blocksXY[1],blocksXY[2],blocksXY[3], filename)

    return newName


def loftrGenerate_area(img0_pth, img1_pth, blocksXY):
    image_pair = [img0_pth, img1_pth]
    matcher = LoFTR(config=default_cfg)

    image_type = 'outdoor'
    if image_type == 'indoor':
        matcher.load_state_dict(torch.load("/weights/indoor_ds.ckpt")['state_dict'])
    elif image_type == 'outdoor':
        matcher.load_state_dict(torch.load("./module/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    else:
        raise ValueError("Wrong image_type is given.")

    # matcher = matcher.eval().cuda()
    matcher = matcher.eval()
    img0_origin = cv2.imread(image_pair[0])
    img1_origin = cv2.imread(image_pair[1])
    img0_origin = cv2.resize(img0_origin, (640, 480))
    img1_origin = cv2.resize(img1_origin, (640, 480))

    '''
        tmp_pth = "/storage/LoFTR/results/originSet/"+dirName
    if not os.path.exists(tmp_pth+returnBaseName(img0_pth)):
        os.makedirs(tmp_pth, exist_ok=True)
        cv2.imwrite(tmp_pth+"/"+returnBaseName(img0_pth), img0_origin) # store resized origin_img0

    '''

    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None] / 255.
    img1 = torch.from_numpy(img1_raw)[None][None] / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    x1, y1, x2, y2 = blocksXY

    # 建立布林 mask，挑出 mkpts0 在指定區域內的點
    mask = (mkpts0[:, 0] >= x1) & (mkpts0[:, 0] <= x2) & \
           (mkpts0[:, 1] >= y1) & (mkpts0[:, 1] <= y2)

    # 用這個 mask 過濾匹配點
    mkpts0_filtered = mkpts0[mask]
    mkpts1_filtered = mkpts1[mask]


    # blocksXY = block_region(mkpts0)
    # blocksXY = generate_random_rectangle(640, 480)
    if (blocksXY == None):
        return

    newName = combine_filenames(returnBaseName(img0_pth), returnBaseName(img1_pth))
    # addWhiteBlock(bloackregion[0],bloackregion[1],bloackregion[2],bloackregion[3],"image0","whiteBlock")

    if len(mkpts0_filtered) >= 4:
        H, status = cv2.findHomography(mkpts0_filtered, mkpts1_filtered, cv2.RANSAC)
    else:
        raise ValueError("Not enough matching points to compute homography.")

    if H is None:
        raise ValueError("Homography estimation failed (H is None).")

    result_img0 = img0_origin.copy()
    target_image1 = img1_origin
    result_img0 = cv2.rectangle(result_img0, (blocksXY[0], blocksXY[1]), (blocksXY[2], blocksXY[3]), (255, 255, 255),
                                -1)
    for dst_y in range(blocksXY[1], blocksXY[3]):
        for dst_x in range(blocksXY[0], blocksXY[2]):
            coordinate = np.array([[dst_x], [dst_y], [1]])
            trans_coordinate = np.dot(H, coordinate)
            # 齊次座標轉二維
            if trans_coordinate[2] == 0 or not np.isfinite(trans_coordinate).all():
                continue
            trans_coordinate = trans_coordinate / trans_coordinate[2]

            src_x, src_y = int(trans_coordinate[0]), int(trans_coordinate[1])
            if ((src_x > 639 or src_x < 0) or (src_y > 479 or src_y < 0)):
                continue
            elif ((dst_x > 639 or dst_x < 0) or (dst_y > 479 or dst_y < 0)):
                continue
            result_img0[dst_y, dst_x] = target_image1[src_y, src_x]  # write pixel

    if not os.path.exists("images060802/origin/"):
        os.makedirs("images060802/origin/", exist_ok=True)

    cv2.imwrite("images060802/origin/" + newName, img0_origin)

    if not os.path.exists("images060802/fixed/"):
        os.makedirs("images060802/fixed/", exist_ok=True)
    filename = "images060802/fixed/" + newName
    # Save the image using OpenCV's imwrite function
    cv2.imwrite(filename, result_img0)
    filename = "images060802/masks/"
    if not os.path.exists("images060802/masks/"):
        os.makedirs("images060802/masks/", exist_ok=True)
    filename += newName
    makeBlackImg(blocksXY[0], blocksXY[1], blocksXY[2], blocksXY[3], filename)

    return newName