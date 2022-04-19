import math
import torch
import argparse
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv import Config

from interpretation.gradcam import GradCAM_YOLOV3, gen_cam

from tqdm import tqdm

# Modefy the label names
label_names = [
    'person bev', 'car bev', 'van bev', 'truck bev', 'bus bev',
    'person', 'car', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'bird', 'camouflage man'
]

def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)

def correspond_box(predictbox, groundtruthboxes):
    iou_ = 0
    index = -1
    for i in range(len(groundtruthboxes)):
        iou__ = iou(predictbox, groundtruthboxes[i])
        if iou__ > iou_:
            iou_ = iou__
            index = i
    if index:
        return groundtruthboxes[index]
    else:
        return False

def maximum_point(mask):
    indexes = np.where(mask==np.max(mask))
    if len(indexes[1]) and len(indexes[0]):
        point = np.array([indexes[1][0], indexes[0][0]])
        return point
    else:
        return False

def point_game(point, gtbox):
    if (point > gtbox[:2]).sum() == 2 and (point < gtbox[:2]).sum() == 2:
        return 1
    else:
        return 0

def parse_args():
    parser = argparse.ArgumentParser(description='YoloV3 Grad-CAM')
    # general
    parser.add_argument('--config',
                        type=str,
                        default = './work_dirs/yolo_v3/yolo_v3.py',
                        help='Yolo V3 configuration.')
    parser.add_argument('--thresh',
                        type=float,
                        default = 0.5,
                        help='Score threshold.')
    parser.add_argument('--checkpoint',
                        type=str,
                        default = './work_dirs/yolo3/yolov3.pth',
                        help='checkpoint.')
    parser.add_argument('--device',
                        type=str,
                        default = 'cuda:0',
                        help='device.')
    parser.add_argument('--save-dir',
                        type=str,
                        default = 'GradCAM/YOLOV3',
                        help='save dir.')

    args = parser.parse_args()
    return args

def main(args):
    # init
    config = args.config
    cfg = Config.fromfile(config)
    checkpoint = args.checkpoint
    device = args.device
    model = init_detector(config, checkpoint, device)

    grad_cam = GradCAM_YOLOV3(model, 'backbone.conv_res_block4.conv.conv')

    # dataset
    dataset = build_dataset(cfg.data.test_PG)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    points_num = 0
    count_num = 0

    # Read the imageset
    for data in tqdm(data_loader):
        image_path = data["img_metas"][0].data[0][0]["filename"]
        image_shape = data["img_metas"][0].data[0][0]['ori_shape']
        image = cv2.imread(image_path)
        scale_factor = data["img_metas"][0].data[0][0]["scale_factor"]
        gt_bboxes = data['gt_bboxes'][0][0]
        gt_bboxes = (gt_bboxes / scale_factor).int()

        ## gradcam
        # for index in range(len(gt_bboxes)):
        for index in range(100):
            mask, box, class_id, score = grad_cam(data, index)
            if score < args.thresh:
                break
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]))

            gt_box = correspond_box(box, gt_bboxes)
            # exist
            if gt_box is not False:
                count_num += 1

                points = maximum_point(mask)
                if points is False:
                    continue

                points_num += point_game(points, gt_box.cpu().numpy())
                
                # image_cam, heatmap = gen_cam(image, mask)
                # draw_image = image_cam.copy()
                # draw_label_type(draw_image, box, gt_box.cpu().numpy(), points, label_names[int(class_id)],line = 5,label_color=(0,255,255))
                # cv2.imwrite("result-"+str(index)+".jpg", draw_image)

    print(points_num, count_num)
        
    return

def draw_label_type(draw_img, bbox, gtbox, points, label, line = 5,label_color=None):
    if label_color == None:
        label_color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]

    gt_color = [0,255,0]

    # label = str(bbox[-1])
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      bbox[:2],
                      bbox[2:],
                      color=label_color,
                      thickness=line)
        cv2.rectangle(draw_img,
                      gtbox[:2],
                      gtbox[2:],
                      color=gt_color,
                      thickness=line)
        cv2.circle(draw_img, points, 5, (0,0,255), 3)
        
    else:
        cv2.rectangle(draw_img,
                      bbox[:2],
                      bbox[2:],
                      color=label_color,
                      thickness=line)
        cv2.rectangle(draw_img,
                      gtbox[:2],
                      gtbox[2:],
                      color=gt_color,
                      thickness=line)
        cv2.circle(draw_img, points, 5, (0,0,255), 3)

if __name__ == "__main__":
    args = parse_args()
    main(args)