# import math
# import torch

import cv2
# import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor

from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv import Config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import random
import argparse
import os
from skimage import io
from utils import mkdir

from interpretation.gradcam import GradCAM_FRCN

np.random.seed(300)

def prepare_img(imgs, model):
    """
    prepare function
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    return data

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    heatmap = cv2.resize(heatmap,
        dsize = (image.shape[1], image.shape[0]))

    # merge heatmap to original image
    # cam = heatmap + np.float32(image)
    return (heatmap * 255).astype(np.uint8)

def plot_cam_image(img, mask, box, class_id, score, bbox_index, COLORS, label_names, save_dir):
    """
    Merge the CAM map to original image
    """
    height, width = img.shape[:2]

    image_tmp = img.copy()
    x1, y1, x2, y2 = box
    # predict_box = img[y1:y2, x1:x2]
    image_heatmap = gen_cam(img[y1:y2, x1:x2], mask)
    image_cam = img[y1:y2, x1:x2]*0.4+image_heatmap*0.6
    
    image_tmp[y1:y2, x1:x2] = image_cam
    image_tmp = cv2.rectangle(image_tmp, (x1,y1), (x2,y2), COLORS[class_id], int(width/112))

    label = label_names[class_id]
    
    cv2.putText(image_tmp, label+": "+"%.2f"%(score*100)+"%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[class_id], 2)

    io.imsave(os.path.join(save_dir, '{}-{}-{}.jpg'.format(bbox_index, class_id, score)), image_tmp)
    
    return None

def main(args):
    # Init your model
    config = args.config
    cfg = Config.fromfile(config)
    checkpoint_path = args.checkpoint
    device = args.device
    model = init_detector(config)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
    label_names = checkpoint['meta']['CLASSES']
    model.CLASSES = checkpoint['meta']['CLASSES']
    COLORS = np.random.uniform(0, 255, size=(len(label_names), 3))

    grad_cam = GradCAM_FRCN(model, 'roi_head.shared_head.layer4.2')

    image = cv2.imread(args.image_path)
    data = prepare_img(image, model)

    ## First is the data, second is the index of the predicted bbox
    mask, box, class_id, score = grad_cam(data, args.bbox_index, "proposal")
    
    org_img = cv2.imread(args.image_path)

    mkdir(args.save_dir)
    plot_cam_image(org_img[..., ::-1], mask, box, class_id, score, args.bbox_index, COLORS, label_names, args.save_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN Grad-CAM')
    # general
    parser.add_argument('--config',
                        type=str,
                        default = 'configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco.py',
                        help='FRCN configuration.')
    parser.add_argument('--checkpoint',
                        type=str,
                        default = 'checkpoints/faster_rcnn_r50_caffe_c4_1x_coco_20220316_150152-3f885b85.pth',
                        help='checkpoint.')
    parser.add_argument('--device',
                        type=str,
                        default = 'cuda:0',
                        help='device.')
    parser.add_argument('--image-path',
                        type=str,
                        default = 'image/cow.png',
                        # default = "/home/cry/data4/Datasets/js-dataset/images/9999962_00000_d_0000088.jpg",
                        help='image path.')
    parser.add_argument('--bbox-index',
                        type=int,
                        default = 0,
                        help='index.')
    parser.add_argument('--save-dir',
                        type=str,
                        default = 'images/GradCAM/FRCN-C4/proposal',
                        help='save dir.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)