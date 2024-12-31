# Simple instructions

Support ODAM and D-RISE.

Detector:

- Mask-RCNN: `mask_rcnn`

- YOLO V3: `yolo_v3`

- SSD: `ssd`

- FCOS: `fcos`

## 1. Environment: 

Please install mmdetection version 3 first.

## 2. For ODAM:

2.1 get `.npy` format saliency maps:

```shell
python traditional_ODAM \
    --Datasets datasets/coco/val2017 \
    --eval-list datasets/coco_ssd_correct.json \
    --detector ssd \
    --save-dir ./baseline_results/tradition-detector-coco-correctly/
```

2.2 compute faithfulness metric:

```shell
python traditional_inference.py \
    --Datasets datasets/coco/val2017 \
    --detector ssd \
    --eval-dir ./baseline_results/tradition-detector-coco-correctly/ssd-ODAM/
```

2.3 visualization:

```shell
python visualize_baseline.py \
    --explanation-dir baseline_results/tradition-detector-coco-correctly/ssd-ODAM \
    --Datasets datasets/coco/val2017
```

## 3. For Grad-CAM:

Just modify ODAM file:

ODAM:

```python
grad = feature.grad    # torch.Size([1, 512, 9, 19])
                
# grad = grad.mean([-1,-2], keepdim=True) # torch.Size([1, 512, 1, 1])
cam_map = F.relu_((grad * feature).sum(1)) # torch.Size([1, 9, 19])
```

Grad-CAM:
```python
grad = feature.grad    # torch.Size([1, 512, 9, 19])
                
grad = grad.mean([-1,-2], keepdim=True) # torch.Size([1, 512, 1, 1])
cam_map = F.relu_((grad * feature).sum(1)) # torch.Size([1, 9, 19])
```
 
## 4. For D-RISE:

2.1 get `.npy` format saliency maps:

```shell
python traditional_DRISE \
    --Datasets datasets/coco/val2017 \
    --eval-list datasets/coco_ssd_correct.json \
    --detector ssd \
    --save-dir ./baseline_results/tradition-detector-coco-correctly/
```

Same instruction 2.2 and 2.3 as ODAM