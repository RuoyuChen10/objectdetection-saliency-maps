# python PointGame-GradCAM.py --config work_dirs/retinanet/retinanet.py \
#     --model retinanet \
#     --checkpoint work_dirs/retinanet/latest.pth \
#     --save-dir results/RetinaNet/

# python PointGame-GradCAM.py --config work_dirs/faster_rcnn-c4/faster_rcnn.py \
#     --model frcn \
#     --checkpoint work_dirs/faster_rcnn-c4/epoch_24.pth \
#     --save-dir results/Faster-RCNN

python PointGame-D-RISE.py --config work_dirs/faster_rcnn-c4/faster_rcnn.py \
    --thresh 0.3 \
    --model frcn \
    --checkpoint work_dirs/faster_rcnn-c4/epoch_24.pth \
    --save-dir results/Faster-RCNN