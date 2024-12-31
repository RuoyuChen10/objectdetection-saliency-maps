import math
import torch

import cv2
# import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor

from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv import Config
import random
import argparse
import os

# Modefy the label names
label_names = [
    'person bev', 'car bev', 'van bev', 'truck bev', 'bus bev',
    'person', 'car', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'bird', 'camouflage man'
]

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
        pred = out[target_class_index]
        score = max([iou(target_box, box) * score for *box, score in pred],
                    default=0)
        res += mask * score
    return res

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
    # print(datas)

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

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

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
    mask = norm_image(mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap

def main(args):
    # Init your model
    config = args.config
    cfg = Config.fromfile(config)
    checkpoint = args.checkpoint
    device = args.device
    model = init_detector(config, checkpoint, device)

    image = cv2.imread(args.image_path)
    data = prepare_img(image, model)

    feat = model.extract_feat(data['img'][0].cuda())
    res = model.bbox_head.simple_test(
        feat, data['img_metas'][0], rescale=True)
    target_box = res[0][0][args.bbox_index][:-1].cpu().detach().numpy().astype(np.int32)


    saliency_map = generate_saliency_map(model,
                                        image,
                                        target_class_index=1,
                                        target_box=target_box,
                                        prob_thresh=0.5,
                                        grid_size=(16, 16),
                                        n_masks=1000)
    
    image_with_bbox = image.copy()

    image_with_bbox, heatmap = gen_cam(image, saliency_map)
    cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 0, 255), 5)

    mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.image_path.split('/')[-1].split(".")[0] + "-bbox-id-" + str(args.bbox_index) + ".jpg")

    cv2.imwrite(save_path, image_with_bbox)

def parse_args():
    parser = argparse.ArgumentParser(description='YoloV3 Grad-CAM')
    # general
    parser.add_argument('--config',
                        type=str,
                        default = 'work_dirs/yolo_v3/yolo_v3.py',
                        help='Yolo V3 configuration.')
    parser.add_argument('--checkpoint',
                        type=str,
                        default = 'work_dirs/yolo_v3/latest.pth',
                        help='checkpoint.')
    parser.add_argument('--device',
                        type=str,
                        default = 'cuda:0',
                        help='device.')
    parser.add_argument('--image-path',
                        type=str,
                        # default = '/home/cry/data4/Datasets/js-dataset/images/0000008_02499_d_0000041.jpg',
                        default = "/home/cry/data4/Datasets/js-dataset/images/9999962_00000_d_0000088.jpg",
                        help='image path.')
    parser.add_argument('--bbox-index',
                        type=int,
                        default = 2,
                        help='index.')
    parser.add_argument('--save-dir',
                        type=str,
                        default = 'DRISE/YOLOV3',
                        help='save dir.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)