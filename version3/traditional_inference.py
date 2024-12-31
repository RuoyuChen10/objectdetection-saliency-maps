import os
import json
import cv2
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

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--steps',
                        type=int,
                        default=49,
                        help='steps.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/coco_yolo_v3_correct.json',
                        help='Datasets.')
    parser.add_argument('--detector',
                        type=str,
                        default='yolo_v3',
                        help='Detector.')
    parser.add_argument('--eval-dir', 
                        type=str, default='./baseline_results/tradition-detector-coco-correctly/yolo_v3-DRISE/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def calculate_iou(batched_boxes, target_box):
    # batched_boxes: [batch, np, 4]
    # target_box: [4]

    # Separation coordinates
    x1, y1, x2, y2 = batched_boxes[..., 0], batched_boxes[..., 1], batched_boxes[..., 2], batched_boxes[..., 3]
    tx1, ty1, tx2, ty2 = torch.tensor(target_box)

    # Calculate intersection area
    inter_x1 = torch.maximum(x1, tx1)
    inter_y1 = torch.maximum(y1, ty1)
    inter_x2 = torch.minimum(x2, tx2)
    inter_y2 = torch.minimum(y2, ty2)

    # 计算相交区域的面积
    inter_area = torch.clamp((inter_x2 - inter_x1), min=0) * torch.clamp((inter_y2 - inter_y1), min=0)

    # Calculate the area of ​​the intersection
    box_area = (x2 - x1) * (y2 - y1)
    target_area = (tx2 - tx1) * (ty2 - ty1)

    # Calculating IoU
    union_area = box_area + target_area - inter_area
    iou = inter_area / union_area

    return iou

def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image.astype(np.uint8)

class Mask_RCNN_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.eps = torch.finfo(torch.float32).eps
        
    def forward(self, image, target_class, target_box):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes+1)
        """
        self.detection_model.zero_grad()
        
        data_ = dict(img=image, img_id=0)
        data_ = self.test_pipeline(data_)
        
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        
        img_shape = data_['data_samples'][0].metainfo['img_shape']
        scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
        data_preproccess = self.detection_model.data_preprocessor(data_, False)
        
        # backbone
        x = self.detection_model.extract_feat(data_preproccess['inputs'])
        
        rpn_results_list = self.detection_model.rpn_head.predict(
                x, data_preproccess['data_samples'], rescale=False)
        
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)
        
        roi_outs = self.detection_model.roi_head.forward(x, rpn_results_list,
                                    data_preproccess['data_samples'])

        bbox_pred = roi_outs[1]

        cls_scores = F.softmax(roi_outs[0], dim=-1)
        
        # bounding boxes
        num_classes = self.detection_model.roi_head.bbox_head.num_classes
        rois = rois.repeat_interleave(num_classes, dim=0)
        bbox_pred = bbox_pred.view(-1, self.detection_model.roi_head.bbox_head.bbox_coder.encode_size)
        bboxes = self.detection_model.roi_head.bbox_head.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        
        bboxes = bboxes.view(-1, num_classes, self.detection_model.roi_head.bbox_head.bbox_coder.encode_size)
        
        bboxes = scale_boxes(bboxes, scale_factor)
        
        target_cls_scores = cls_scores[:, target_class] # torch.Size([1000])
        target_predict_bboxes = bboxes[:, target_class] # torch.Size([1000, 4])
        
        ious = calculate_iou(target_predict_bboxes, target_box) # torch.Size([1000])
        
        target_scores = ious *  target_cls_scores
        idx = target_scores.argmax()
        
        object_score = target_scores[idx]
        iou = ious[idx]
        target_class_score = target_cls_scores[idx]
        predict_box = bboxes[idx]
        
        return {
            "object_score": object_score.cpu().item(),
            "predict_box": predict_box.cpu().numpy().astype(int).tolist(),
            "predict_cls_score": target_class_score.cpu().item(),
            "iou": iou.cpu().item()
        }
        
class YOLO_V3_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.eps = torch.finfo(torch.float32).eps
        
    def forward(self, image, target_class, target_box):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes)
        """
        self.detection_model.zero_grad()
        
        data_ = dict(img=image, img_id=0)
        data_ = self.test_pipeline(data_)
        
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        
        # with torch.no_grad():
        data_preproccess = self.detection_model.data_preprocessor(data_, False)
        scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
        
        x = self.detection_model.extract_feat(data_preproccess['inputs'])
        pred_maps = self.detection_model.bbox_head.forward(x)[0]
        
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]
        mlvl_anchors = self.detection_model.bbox_head.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.detection_model.bbox_head.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(1, -1,
                                                    self.detection_model.bbox_head.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(
                pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4] # torch.Size([1, 17955, 4])
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid() # torch.Size([1, 17955, 80])
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.detection_model.bbox_head.bbox_coder.decode(flatten_anchors,
            flatten_bbox_preds,
            flatten_strides.unsqueeze(-1))  # 不定长，可以选top 400
        bboxes = scale_boxes(flatten_bboxes, scale_factor)

        ious = calculate_iou(bboxes, target_box)
        
        target_scores = ious *  flatten_cls_scores[:, :, target_class]
        idx = target_scores.argmax()

        object_score = target_scores[0, idx]
        iou = ious[0, idx]
        target_class_score = flatten_cls_scores[0, idx, target_class]
        predict_box = bboxes[0,idx]

        return {
            "object_score": object_score.cpu().item(),
            "predict_box": predict_box.cpu().numpy().astype(int).tolist(),
            "predict_cls_score": target_class_score.cpu().item(),
            "iou": iou.cpu().item()
        }

class FCOS_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.top_k = 2000
        self.eps = torch.finfo(torch.float32).eps
        
    def forward(self, image, target_class, target_box):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes)
        """
        self.detection_model.zero_grad()
        
        data_ = dict(img=image, img_id=0)
        data_ = self.test_pipeline(data_)
        
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        
        data_preproccess = self.detection_model.data_preprocessor(data_, False)
        scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
        
        x = self.detection_model.extract_feat(data_preproccess['inputs'])
        # for x_ in x:
        #     x_.retain_grad()
        
        cls_scores, bbox_preds, score_factors = self.detection_model.bbox_head.forward(x)
        
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.detection_model.bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        
        cls_score_list = select_single_mlvl(
            cls_scores, 0, detach=False)
        bbox_pred_list = select_single_mlvl(
            bbox_preds, 0, detach=False)
        score_factor_list = select_single_mlvl(
            score_factors, 0, detach=False)
            
        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
        enumerate(zip(cls_score_list, bbox_pred_list,
                        score_factor_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        
            dim = self.detection_model.bbox_head.bbox_coder.encode_size
            
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            score_factor = score_factor.permute(1, 2,
                                                0).reshape(-1).sigmoid()
            
            cls_score = cls_score.permute(1, 2,
                                        0).reshape(-1, self.detection_model.bbox_head.cls_out_channels)
            scores = cls_score.sigmoid()
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_valid_priors.append(priors)
            
        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
    
        bboxes = self.detection_model.bbox_head.bbox_coder.decode(priors, bbox_pred, max_shape=data_['data_samples'][0].metainfo["img_shape"])
        scores = torch.cat(mlvl_scores)
        
        bboxes = scale_boxes(bboxes, scale_factor)
        
        ious = calculate_iou(bboxes, target_box)
        
        target_scores = ious * scores[:, target_class]
        idx = target_scores.argmax()

        object_score = target_scores[idx]
        iou = ious[idx]
        target_class_score = scores[idx, target_class]
        predict_box = bboxes[idx]

        return {
            "object_score": object_score.cpu().item(),
            "predict_box": predict_box.cpu().numpy().astype(int).tolist(),
            "predict_cls_score": target_class_score.cpu().item(),
            "iou": iou.cpu().item()
        }

class SSD_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.top_k = 2000
        self.eps = torch.finfo(torch.float32).eps
        
    def forward(self, image, target_class, target_box):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes)
        """
        self.detection_model.zero_grad()
        
        data_ = dict(img=image, img_id=0)
        data_ = self.test_pipeline(data_)
        
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        
        data_preproccess = self.detection_model.data_preprocessor(data_, False)
        scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
        
        x = self.detection_model.extract_feat(data_preproccess['inputs'])
        # for x_ in x:
        #     x_.retain_grad()
            
        cls_scores, bbox_preds = self.detection_model.bbox_head.forward(x)
        
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.detection_model.bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        
        cls_score_list = select_single_mlvl(
            cls_scores, 0, detach=False)
        bbox_pred_list = select_single_mlvl(
            bbox_preds, 0, detach=False)
        score_factor_list = [None for _ in range(num_levels)]
        # score_factor_list = select_single_mlvl(
        #     score_factors, 0, detach=True)
            
        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
        enumerate(zip(cls_score_list, bbox_pred_list,
                        score_factor_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        
            dim = self.detection_model.bbox_head.bbox_coder.encode_size
            
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            # score_factor = score_factor.permute(1, 2,
            #                                     0).reshape(-1).sigmoid()
            
            cls_score = cls_score.permute(1, 2,
                                        0).reshape(-1, self.detection_model.bbox_head.cls_out_channels)
            scores = cls_score.softmax(-1)[:, :-1]
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_valid_priors.append(priors)
            
        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        
        bboxes = self.detection_model.bbox_head.bbox_coder.decode(priors, bbox_pred, max_shape=data_['data_samples'][0].metainfo["img_shape"])
        scores = torch.cat(mlvl_scores)
        
        bboxes = scale_boxes(bboxes, scale_factor)
        
        ious = calculate_iou(bboxes, target_box)
        
        target_scores = ious *  scores[:, target_class]
        idx = target_scores.argmax()
        
        object_score = target_scores[idx]
        iou = ious[idx]
        target_class_score = scores[idx, target_class]
        predict_box = bboxes[idx]
        
        return {
            "object_score": object_score.cpu().item(),
            "predict_box": predict_box.cpu().numpy().astype(int).tolist(),
            "predict_cls_score": target_class_score.cpu().item(),
            "iou": iou.cpu().item()
        }
        

def main(args):
    device = "cuda"
    assert args.detector in args.eval_list
    assert args.detector in args.eval_dir
    
    if args.detector == "mask_rcnn":
        config = 'config/mask-rcnn_r50_fpn_2x_coco.py'
        checkpoint = 'ckpt/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
        model = init_detector(config, checkpoint, device)
        detection_model = Mask_RCNN_R(model.eval(), device)
    elif args.detector == "yolo_v3":
        config = 'config/yolov3_d53_8xb8-ms-608-273e_coco.py'
        checkpoint = 'ckpt/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
        model = init_detector(config, checkpoint, device)
        detection_model = YOLO_V3_R(model.eval(), device)
    elif args.detector == "fcos":
        config = 'config/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py'
        checkpoint = 'ckpt/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
        model = init_detector(config, checkpoint, device)
        detection_model = FCOS_R(model.eval(), device)
    elif args.detector == "ssd":
        config = "config/ssd300_coco.py"
        checkpoint = "ckpt/ssd300_coco_20210803_015428-d231a06e.pth"
        model = init_detector(config, checkpoint, device)
        model.bbox_head.loss_cls = False
        detection_model = SSD_R(model.eval(), device)
        
    json_save_dir = os.path.join(args.eval_dir, "json")
    mkdir(json_save_dir)
    npy_dir = os.path.join(args.eval_dir, "npy")
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
        
    select_infos = val_file["case1"]
    for info in tqdm(select_infos[:]):
        if os.path.exists(
            os.path.join(json_save_dir, info["file_name"].replace(".jpg", "_{}.json").format(info["id"]))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, info["file_name"])
        image = cv2.imread(image_path)
        
        saliency_map = np.load(
                os.path.join(npy_dir, info["file_name"].replace(".jpg", "_{}.npy").format(info["id"]))
            )
        
        target_box = info["bbox"]
        target_label = coco_classes.index(info["category"])
        
        json_file = {}
        json_file["insertion_score"] = []
        json_file["deletion_score"] = []
        json_file["insertion_iou"] = []
        json_file["insertion_box"] = []
        json_file["insertion_cls"] = []
        json_file["deletion_iou"] = []
        json_file["deletion_box"] = []
        json_file["deletion_cls"] = []
        json_file["region_area"] = []
        json_file["target_box"] = target_box
        json_file["target_label"] = info["category"]
        
        for i in range(1, args.steps+1):
            perturbed_rate = i / args.steps
            json_file["region_area"].append(perturbed_rate)
            
            # insertion
            insertion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "insertion")
            
            with torch.no_grad():
                results = detection_model(insertion_image, target_label, target_box)
            
            json_file["insertion_score"].append(results["object_score"])
            json_file["insertion_iou"].append(results["iou"])
            json_file["insertion_box"].append(results["predict_box"])
            json_file["insertion_cls"].append(results["predict_cls_score"])
            
            # deletion
            deletion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "deletion")
            
            with torch.no_grad():
                results = detection_model(deletion_image, target_label, target_box)
                
            json_file["deletion_score"].append(results["object_score"])
            json_file["deletion_iou"].append(results["iou"])
            json_file["deletion_box"].append(results["predict_box"])
            json_file["deletion_cls"].append(results["predict_cls_score"])
            
        with open(
                os.path.join(json_save_dir, info["file_name"].replace(".jpg", "_{}.json").format(info["id"])), "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        # break
    
if __name__ == "__main__":
    args = parse_args()
    main(args)