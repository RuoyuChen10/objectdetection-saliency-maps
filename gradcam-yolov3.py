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
import random
import argparse
import os

from interpretation.gradcam import GradCAM_YOLOV3

# Modefy the label names
label_names = [
    'person bev', 'car bev', 'van bev', 'truck bev', 'bus bev',
    'person', 'car', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'bird', 'camouflage man'
]

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
    print(datas)

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
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap

def draw_label_type(draw_img,bbox,label, line = 5,label_color=None):
    if label_color == None:
        label_color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]

    # label = str(bbox[-1])
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      bbox[:2],
                      bbox[2:],
                      color=label_color,
                      thickness=line)
    else:
        cv2.rectangle(draw_img,
                      bbox[:2],
                      bbox[2:],
                      color=label_color,
                      thickness=line)

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def main(args):
    # Init your model
    config = args.config
    cfg = Config.fromfile(config)
    checkpoint = args.checkpoint
    device = args.device
    model = init_detector(config, checkpoint, device)

    grad_cam = GradCAM_YOLOV3(model, 'backbone.conv_res_block4.conv.conv')

    image = cv2.imread(args.image_path)
    data = prepare_img(image, model)

    ## First is the data, second is the index of the predicted bbox
    mask, box, class_id = grad_cam(data, args.bbox_index)

    # rendering
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    image_cam, heatmap = gen_cam(image, mask)

    draw_image = image_cam.copy()
    draw_label_type(draw_image,box,label_names[int(class_id)],line = 5,label_color=(0,0,255))

    mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.image_path.split('/')[-1].split(".")[0] + "-bbox-id-" + str(args.bbox_index) + ".jpg")

    cv2.imwrite(save_path, draw_image)

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
                        default = 'GradCAM/YOLOV3',
                        help='save dir.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)