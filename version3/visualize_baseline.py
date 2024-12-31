# -*- coding: utf-8 -*-

"""
Created on 2024/10/11

@author: Ruoyu Chen
"""

import argparse

import scipy
import os
import cv2
import json
import imageio
import numpy as np
from PIL import Image
import supervision as sv
import torch

import matplotlib
from matplotlib import pyplot as plt

from torchvision.ops import box_convert

from tqdm import tqdm
from utils import *

matplotlib.get_cachedir()
plt.rc('font', family="Arial")

from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Faithfulness Metric')
    parser.add_argument('--explanation-dir', 
                        type=str, 
                        default='baseline_results/tradition-detector-coco-correctly/mask_rcnn-ODAM',
                        help='Save path for saliency maps generated by our methods.')
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco/val2017',
                        help='Datasets.')
    args = parser.parse_args()
    return args

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

def gen_cam(image_path, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image
    w = mask.shape[1]
    h = mask.shape[0]
    image = cv2.resize(cv2.imread(image_path), (w,h))
    # mask->heatmap
    mask = cv2.resize(mask, (int(w/20),int(h/20)))
    mask = cv2.resize(mask, (w,h))
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_COOL)  # cv2.COLORMAP_COOL
    heatmap = np.float32(heatmap)

    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam.astype(np.uint8), (heatmap).astype(np.uint8)

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def annotate_with_grounding_dino(image, boxes, phrases, color=(34,139,34)):
    """
    使用 Grounding DINO 格式化可视化目标检测结果

    参数:
        image (np.ndarray): 输入图像 (BGR) 格式
        boxes (np.ndarray): 检测框坐标，格式为 xyxy，形状为 (N, 4)
        phrases (List[str]): 每个检测框对应的类别标签列表

    返回:
        np.ndarray: 可视化的图像
    """
    # 将坐标转换为 Torch 张量，并确保数据类型一致
    boxes = torch.tensor(boxes, dtype=torch.float32)
    
    class_ids = np.zeros(len(boxes), dtype=int)
    
    # 获取图像的宽和高
    h, w, _ = image.shape

    # 确保坐标与图像尺寸匹配
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)  # 限制x坐标在图像范围内
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)  # 限制y坐标在图像范围内

    # 将框转换为 cxcywh 格式，再转换回 xyxy 格式
    xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
    
    # 使用 supervision 库来进行可视化
    detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)

    # 初始化监督库中的注释器
    bbox_annotator = sv.BoxAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
    label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))

    # 转换图像格式为 BGR（OpenCV 格式）
    # annotated_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_frame = image
    
    # 绘制边框
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # 绘制标签
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)

    return annotated_frame

def visualization(image, attribution_map, saved_json_file, vis_image, class_name="object", index=None, mode="insertion"):
    # S_set_add = S_set.copy()
    # S_set_add = np.array([S_set_add[0]-S_set_add[0]] + S_set_add)
    # image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    image_baseline = image.copy()
    
    insertion_ours_images = [perturbed(image, attribution_map, rate = 0, mode = "insertion")]
    deletion_ours_images = [perturbed(image, attribution_map, rate = 0, mode = "deletion")]

    for perturbed_rate in saved_json_file["region_area"]:
        insertion_ours_images.append(
            perturbed(image, attribution_map, rate = perturbed_rate, mode = "insertion")
        )
        deletion_ours_images.append(
            perturbed(image, attribution_map, rate = perturbed_rate, mode = "deletion")
        )
    
    # insertion_image = (S_set[0] - S_set[0]) * image_baseline
    # insertion_ours_images.append(insertion_image)
    # deletion_ours_images.append(image_baseline)
    # for smdl_sub_mask in S_set:
    #     insertion_image = insertion_image.copy() + smdl_sub_mask * image_baseline
    #     insertion_ours_images.append(insertion_image)
    #     deletion_ours_images.append(image_baseline - insertion_image)

    if mode == "insertion":
        curve_score = [saved_json_file["deletion_score"][-1]] + saved_json_file["insertion_score"]
        # curve_cls_score = [saved_json_file["deletion_cls"][-1]] + saved_json_file["insertion_cls"]
        # curve_iou_score = [saved_json_file["deletion_iou"][-1]] + saved_json_file["insertion_iou"]
    elif mode == "deletion":
        curve_score = [saved_json_file["insertion_score"][-1]] + saved_json_file["deletion_score"]

    if index == None:
        if mode == "insertion":
            # ours_best_index = np.argmax(np.array(curve_cls_score) * (np.array(curve_iou_score)>0.75))
            ours_best_index = np.argmax(curve_score)
        elif mode == "deletion":
            ours_best_index = np.argmin(curve_score)
    else:
        ours_best_index = index
    x = [0.0] + saved_json_file["region_area"]
    i = len(x)

    # fig, [ax1, ax2, ax3] = plt.subplots(1,3, gridspec_kw = {'width_ratios':[2, 2, 1.5]}, figsize=(30,8))
    # fig.subplots_adjust(wspace=0.4)  # 增加wspace值，增加图2和图3之间的间隔
    fig = plt.figure(figsize=(30, 8))
    
    ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])  # 图1位置
    ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])  # 图2位置
    ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title('Attribution Map', fontsize=54)
    ax1.set_facecolor('white')
    ax1.imshow(vis_image[...,::-1].astype(np.uint8))
    
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.xaxis.set_visible(True)
    ax2.yaxis.set_visible(False)
    ax2.set_title('Searched Region', fontsize=54)
    ax2.set_facecolor('white')
    ax2.set_xlabel("Object Score: {:.2f}".format(curve_score[ours_best_index]), fontsize=44)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax3.set_xlim((0, 1))
    ax3.set_ylim((0, 1))
    yticks = ax3.get_yticks()
    yticks = yticks[yticks != 0]
    ax3.set_yticks(yticks)
    
    ax3.set_ylabel('Object Score', fontsize=44)
    if mode == "insertion":
        ax3.set_xlabel('Percentage of image revealed', fontsize=44)
    elif mode == "deletion":
        ax3.set_xlabel('Percentage of image removed', fontsize=44)
    ax3.tick_params(axis='both', which='major', labelsize=36)

    if mode == "insertion":
        curve_color = "#FF4500"
    elif mode == "deletion":
        curve_color = "#1E90FF"

    x_ = x[:i]
    ours_y = curve_score[:i]
    ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5)  # draw curve
    ax3.set_facecolor('white')
    ax3.spines['bottom'].set_color('black')
    ax3.spines['bottom'].set_linewidth(2.0)
    ax3.spines['top'].set_color('none')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(2.0)
    ax3.spines['right'].set_color('none')

    # plt.legend(["Ours"], fontsize=40, loc="upper left")
    ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)  # Plot latest point
    # 在曲线下方填充淡蓝色
    ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)

    kernel = np.ones((10, 10), dtype=np.uint8)
    # ax3.plot([x_[ours_best_index], x_[ours_best_index]], [0, 1], color='red', linewidth=3.5)  # 绘制红色曲线
    ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)  # 绘制红色垂直线

    # Ours
    if mode == "insertion":
        mask = ((image - insertion_ours_images[ours_best_index]).sum(-1)>0).astype('uint8')
    elif mode == "deletion":
        mask = 1-((image - insertion_ours_images[ours_best_index]).sum(-1)>0).astype('uint8')

    if ours_best_index != 0:
        dilate = cv2.dilate(mask, kernel, 3)
        # erosion = cv2.erode(dilate, kernel, iterations=3)
        # dilate = cv2.dilate(erosion, kernel, 2)
        edge = dilate - mask
        # erosion = cv2.erode(dilate, kernel, iterations=1)

    image_debug = image_baseline.copy()
    image_debug[mask>0] = image_debug[mask>0] * 0.3
    if ours_best_index != 0:
        image_debug[edge>0] = np.array([0,0,255])
    
    if mode == "insertion":
        if ours_best_index != 0:
            target_box = saved_json_file["insertion_box"][ours_best_index-1]
            cls_score = saved_json_file["insertion_cls"][ours_best_index-1]
        else:
            target_box = saved_json_file["deletion_box"][-1]
            cls_score = saved_json_file["deletion_cls"][-1]
        color=(255,69,0)
    elif mode == "deletion":
        if ours_best_index != 0:
            target_box = saved_json_file["deletion_box"][ours_best_index-1]
            cls_score = saved_json_file["deletion_cls"][ours_best_index-1]
        else:
            target_box = saved_json_file["insertion_box"][-1]
            cls_score = saved_json_file["insertion_cls"][-1]
        color=(30,144,255)
    image_debug = cv2.resize(image_debug, (image.shape[1],image.shape[0]))
    
    if len(target_box) == 80:
        label = coco_classes.index(saved_json_file["target_label"])
        target_box = target_box[label]
    
    image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), ["{}: {:.2f}".format(class_name, cls_score)], color)
    # image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), ["{}".format(class_name)], color)
    ax2.imshow(image_debug[...,::-1])
    
    auc = metrics.auc(x, curve_score)
    if mode == "insertion":
        ax3.set_title('Insertion {:.4f}'.format(auc), fontsize=54)
    elif mode == "deletion":
        ax3.set_title('Deletion {:.4f}'.format(auc), fontsize=54)

def main(args):
    print(args.explanation_dir)
    
    json_root_file = os.path.join(args.explanation_dir, "json")
    npy_root_file = os.path.join(args.explanation_dir, "npy")
    
    full_visualization_path = os.path.join(args.explanation_dir, "full_visualization")
    full_visualization_insertion_path = os.path.join(full_visualization_path, "insertion")
    full_visualization_deletion_path = os.path.join(full_visualization_path, "deletion")
    mkdir(full_visualization_insertion_path)
    mkdir(full_visualization_deletion_path)
        
    json_file_names = os.listdir(json_root_file)
    for json_file_name in tqdm(json_file_names):
        json_file_path = os.path.join(json_root_file, json_file_name)
        npy_file_path = os.path.join(npy_root_file, json_file_name.replace(".json", ".npy"))
        if "refcoco" in args.Datasets:
            image_path = os.path.join(args.Datasets, 
                                      json_file_name.replace("_"+json_file_name.split("_")[-1], ".jpg"))
        elif "lvis" in args.Datasets:
            image_path = os.path.join(args.Datasets, json_file_name.split("_")[0] + "/" + json_file_name.split("_")[1]+".jpg")
        else:
            image_path = os.path.join(args.Datasets, json_file_name.split("_")[0]+".jpg")
        
        save_full_visualization_map_insertion_path = os.path.join(full_visualization_insertion_path, json_file_name.replace(".json", ".png"))
        save_full_visualization_map_deletion_path = os.path.join(full_visualization_deletion_path, json_file_name.replace(".json", ".png"))
                
        if os.path.exists(save_full_visualization_map_insertion_path) and os.path.exists(save_full_visualization_map_deletion_path):
            continue
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            saved_json_file = json.load(f)
        
        # if "_132" not in json_file_name:
        #     continue
        
        attribution_map = np.load(npy_file_path)
        
        image = cv2.imread(image_path)
        
        # attribution_map, _ = add_value(S_set, saved_json_file)
        attribution_map = np.load(npy_file_path)
        
        vis_saliency_map, heatmap = gen_cam(image_path, norm_image(attribution_map))
        vis_saliency_map = cv2.resize(vis_saliency_map, (image.shape[1], image.shape[0]))
        
        # S_set = [perturbed(image, attribution_map, rate = perturbed_rate, mode = "insertion") for perturbed_rate in saved_json_file["region_area"]]
        # S_set = np.array(S_set)
        # S_set[1:] = S_set[1:] - S_set[:-1]
        
        target_box = saved_json_file["target_box"]
        
        # for category in coco_classes:
        #     if target_label == coco_classes_grounding_idx[category]:
        #         target_category = category
        #         break
        if "refcoco" in args.Datasets:
            target_category = saved_json_file["category"]
        else:
            target_category = saved_json_file["target_label"]
        
        vis_saliency_map_w_box = annotate_with_grounding_dino(vis_saliency_map, np.array([target_box]), [target_category])
        
        visualization(image, attribution_map, saved_json_file, vis_saliency_map_w_box, target_category)
        plt.savefig(save_full_visualization_map_insertion_path, bbox_inches='tight',pad_inches=0.0)
        plt.clf()
        plt.close()
        
        visualization(image, attribution_map, saved_json_file, vis_saliency_map_w_box, target_category, mode="deletion")
        plt.savefig(save_full_visualization_map_deletion_path, bbox_inches='tight',pad_inches=0.0)
        plt.clf()
        plt.close()
        
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)