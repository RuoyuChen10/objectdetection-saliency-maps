# Objectdetection Saliency Maps

Based on [mmdetection](https://github.com/open-mmlab/mmdetection) framework. You need to install MMDetaction first, follow here: [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)

## 1. Grad-CAM

> Selvaraju, Ramprasaath R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." International Journal of Computer Vision 128.2 (2020): 336-359.

Paper Url: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)

![](images/GradCAM/Grad-CAM.png)

Supported Object Detection Algorithm:


<details>
<summary>Yolo V3</summary>

Paper: [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)

Step by step see: [gradcam-yolov3.ipynb](tutorial/gradcam-yolov3.ipynb)

```angular2html
python gradcam-yolov3.py \
        --config <Configs Path> \
        --checkpoint <Checkpoint Path> \
        --image-path <Your Image Path> \
        --bbox-index 0 \
        --save-dir images/GradCAM/YOLOV3
```

Visualization:

| ![](images/GradCAM/YOLOV3/0000008_02499_d_0000041-bbox-id-0.jpg)  | ![](images/GradCAM/YOLOV3/0000008_02499_d_0000041-bbox-id-1.jpg) | ![](images/GradCAM/YOLOV3/0000008_02499_d_0000041-bbox-id-2.jpg) |
|  ----  | ----  | ----  |
| ![](images/GradCAM/YOLOV3/9999962_00000_d_0000088-bbox-id-0.jpg) | ![](images/GradCAM/YOLOV3/9999962_00000_d_0000088-bbox-id-1.jpg) | ![](images/GradCAM/YOLOV3/9999962_00000_d_0000088-bbox-id-2.jpg) |


</details>

## 2. D-RISE

> Vitali Petsiuk, Rajiv Jain, Varun Manjunatha, Vlad I. Morariu, Ashutosh Mehra, Vicente Ordonez, Kate Saenko; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 11443-11452

Paper Url: [https://openaccess.thecvf.com/content/CVPR2021/html/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.html](https://openaccess.thecvf.com/content/CVPR2021/html/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.html)

![](images/DRISE/DRISE.png)

Supported Object Detection Algorithm:


<details>
<summary>Yolo V3</summary>

Paper: [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)

Step by step see: [drise-yolov3.ipynb](tutorial/drise-yolov3.ipynb)

```angular2html
python drise-yolov3.py \
        --config <Configs Path> \
        --checkpoint <Checkpoint Path> \
        --image-path <Your Image Path> \
        --bbox-index 0 \
        --save-dir images/DRISE/YOLOV3
```

Visualization:

| ![](images/DRISE/YOLOV3/0000008_02499_d_0000041-bbox-id-0.jpg)  | ![](images/DRISE/YOLOV3/0000008_02499_d_0000041-bbox-id-1.jpg) | ![](images/DRISE/YOLOV3/0000008_02499_d_0000041-bbox-id-2.jpg) |
|  ----  | ----  | ----  |
| ![](images/DRISE/YOLOV3/9999962_00000_d_0000088-bbox-id-0.jpg) | ![](images/DRISE/YOLOV3/9999962_00000_d_0000088-bbox-id-1.jpg) | ![](images/DRISE/YOLOV3/9999962_00000_d_0000088-bbox-id-2.jpg) |


</details>