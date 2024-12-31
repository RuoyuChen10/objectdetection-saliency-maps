import os
import json
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from sklearn import metrics
plt.rc('font', family="Arial")

import torch
import torch.nn.functional as F

from torchvision.ops import box_convert
from utils import *

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
from mmdet.structures.bbox import (bbox2roi,cat_boxes, get_box_tensor, get_box_wh, scale_boxes)
from mmdet.models.utils import select_single_mlvl, filter_scores_and_topk
from mmengine import Config
import mmcv

import argparse
from tqdm import tqdm

def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask

def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked

def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)

def generate_saliency_map(model,
                          image,
                          target_class_index,
                          target_box,
                          prob_thresh=0.5,
                          grid_size=(16, 16),
                          n_masks=5000,
                          seed=0):
    np.random.seed(seed)
    image_h, image_w = image.shape[:2]
    res = np.zeros((image_h, image_w), dtype=np.float32)
    for _ in range(n_masks):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)
        masked = mask_image(image, mask)
        out = inference_detector(model, masked)
        
        bboxes = out.pred_instances.bboxes
        scores = out.pred_instances.scores * (out.pred_instances.labels == target_class_index)

        map_score = max([iou(target_box, box.cpu()) * score.cpu() for box, score in zip(bboxes,scores)],
                    default=0)
        if len(scores) != 0:
            map_score = map_score.item()
        res += mask * map_score
    return res

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/coco_ssd_correct.json',
                        help='Datasets.')
    parser.add_argument('--detector',
                        type=str,
                        default='ssd',
                        help='Detector.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/tradition-detector-coco-correctly/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def main(args):
    device = "cuda"
    assert args.detector in args.eval_list
    
    if args.detector == "mask_rcnn":
        config = 'config/mask-rcnn_r50_fpn_2x_coco.py'
        checkpoint = 'ckpt/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
        model = init_detector(config, checkpoint, device)
    elif args.detector == "yolo_v3":
        config = 'config/yolov3_d53_8xb8-ms-608-273e_coco.py'
        checkpoint = 'ckpt/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
        model = init_detector(config, checkpoint, device)
    elif args.detector == "fcos":
        config = 'config/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py'
        checkpoint = 'ckpt/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
        model = init_detector(config, checkpoint, device)
    elif args.detector == "ssd":
        config = "config/ssd300_coco.py"
        checkpoint = "ckpt/ssd300_coco_20210803_015428-d231a06e.pth"
        model = init_detector(config, checkpoint, device)
        model.bbox_head.loss_cls = False
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
        
    save_dir = os.path.join(
        os.path.join(args.save_dir, "{}-DRISE".format(args.detector)), "npy")
    mkdir(save_dir)
    
    select_infos = val_file["case1"]
    for info in tqdm(select_infos[:]):
        if os.path.exists(
            os.path.join(save_dir, info["file_name"].replace(".jpg", "_{}.npy".format(info["id"])))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, info["file_name"])
        image = cv2.imread(image_path)
        
        target_box = info["bbox"]
        h,w = image.shape[:2]
        target_label = coco_classes.index(info["category"])
        
        saliency_map = generate_saliency_map(model,
                                        image,
                                        target_class_index=target_label,
                                        target_box=target_box,
                                        prob_thresh=0.5,
                                        grid_size=(16, 16),
                                        n_masks=1000)

        np.save(os.path.join(save_dir, info["file_name"].replace(".jpg", "_{}.npy").format(info["id"])), saliency_map)
        
if __name__ == "__main__":
    args = parse_args()
    
    main(args)