import numpy as np
import torch
import cv2
import random
import torch.nn.functional as F

from mmdet.core.utils import select_single_mlvl
from mmdet.core import (bbox2roi, multiclass_nms)

class GradCAM_RetinaNet(object):
    """
    Grad CAM for RetinaNet in mmdetection framework

    查清楚bbox_head.get_bboxes这里

    simple_test
        simple_test_bboxes
            self.forward()
            self.get_bboxes
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Rewriten version
        """
        assert len(cls_scores) == len(bbox_preds)
        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.net.bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=False)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=False)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id, detach=False)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self.net.bbox_head._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def __call__(self, data, index=0):
        """
        :param image: cv2 format, single image
        :param index: Which bounding box
        :return:
        """
        self.net.zero_grad()
        # Important
        feat = self.net.extract_feat(data['img'][0].cuda())

        if type(data['img_metas'][0]) == list:
            img_metas = data['img_metas'][0]
        else:
            img_metas = data['img_metas'][0].data[0]

        # res = self.net.bbox_head.simple_test(
        #     feat, img_metas, rescale=True)
        outs = self.net.bbox_head.forward(feat)
        res = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=True)
        
        if res[0][0].shape[0] == 0:
            return None, None, None, None
        
        score = res[0][0][index][4]
        score.backward()

        gradient = self.gradient[0] # [C,H,W]
        weight = torch.mean(gradient, axis=(1, 2))  # [C]
        feature = self.feature[0]  # [C,H,W]

        # print(gradient.shape, weight.shape, feature.shape)
        
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)  # ReLU

        # Normalization
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        # resize to 224*224
        box = res[0][0][index][:-1].cpu().detach().numpy().astype(np.int32)
        
        class_id = res[0][1][index].cpu().detach().numpy()
        return cam.cpu().detach().numpy(), box, class_id, score.cpu().detach().numpy()

class GradCAM_YOLOV3(object):
    """
    Grad CAM for Yolo V3 in mmdetection framework
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, data, index=0):
        """
        :param image: cv2 format, single image
        :param index: Which bounding box
        :return:
        """
        self.net.zero_grad()
        # Important
        feat = self.net.extract_feat(data['img'][0].cuda())

        if type(data['img_metas'][0]) == list:
            img_metas = data['img_metas'][0]
        else:
            img_metas = data['img_metas'][0].data[0]

        res = self.net.bbox_head.simple_test(
            feat, img_metas, rescale=True)
        score = res[0][0][index][4]
       
        score.backward()

        gradient = self.gradient[0] # [C,H,W]
        weight = torch.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0]  # [C,H,W]

        # print(gradient.shape, weight.shape, feature.shape)
        
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)  # ReLU

        # Normalization
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        # resize to 224*224
        box = res[0][0][index][:-1].cpu().detach().numpy().astype(np.int32)
        
        class_id = res[0][1][index].cpu().detach().numpy()
        return cam.cpu().detach().numpy(), box, class_id, score.cpu().detach().numpy()

# GPU version
class GradCAM_FRCN(object):
    """
    Grad CAM for Faster R-CNN C4 in mmdetection framework
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
    
    def rpn_get_bboxes(self, cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=False,
                   **kwargs):
        assert len(cls_scores) == len(bbox_preds)
        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.net.rpn_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=False)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=False)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id, detach=False)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            results = self.net.rpn_head._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.
        This function needn't read.
        """
        rois = bbox2roi(proposals)
        # print("rois: {}".format(rois.shape))
        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.net.roi_head.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self.net.roi_head._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.net.roi_head.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.net.roi_head.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label, det_inds = self.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels, det_inds
    
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        
        scores = F.softmax(
            cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.net.roi_head.bbox_head.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, inds = multiclass_nms(bboxes, scores,     # return_inds=True
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds=True)
            return det_bboxes, det_labels, inds

    def __call__(self, data, index=0, mode = "proposal"):
        """
        :param image: cv2 format, single image
        :param index: Which bounding box
        "param mode: [proposal, global]
        :return:
        """
        self.net.zero_grad()
        # Important
        feat = self.net.extract_feat(data['img'][0].cuda())

        if type(data['img_metas'][0]) == list:
            img_metas = data['img_metas'][0]
        else:
            img_metas = data['img_metas'][0].data[0]
        
        if mode is "proposal":
            rpn_outs = self.net.rpn_head(feat)
            proposal_list = self.rpn_get_bboxes(*rpn_outs, img_metas=img_metas)
        # print(proposal_list[0].shape)
        # proposal_list = model.rpn_head.simple_test_rpn(feat, img_metas)
        # res = model.roi_head.simple_test(feat, proposal_list, img_metas, rescale=True)
            res = self.simple_test_bboxes(feat, img_metas, proposal_list, self.net.roi_head.test_cfg, rescale=True)
        
            ind = int(res[2][index]/len(self.net.CLASSES))
        elif mode is "global":
            rpn_outs = self.net.rpn_head(feat)
            proposal_list = self.rpn_get_bboxes(*rpn_outs, img_metas=img_metas)
            res= self.net.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.net.roi_head.test_cfg, rescale=True)
        
        score = res[0][0][index][4]
        score.backward()
        
        if mode is "proposal":
            gradient = self.gradient[ind]  # [C,H,W]
            weight = torch.mean(gradient, axis=(1, 2))  # [C]
            feature = self.feature[ind]  # [C,H,W]

        elif mode is "global":
            gradient = self.gradient[0]    # [C,H,W]
            weight = torch.mean(gradient, axis=(1, 2))  # [C]
            feature = self.feature[0]      # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)  # ReLU

        # Normalization
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        # resize to 224*224
        box = res[0][0][index][:-1].cpu().detach().numpy().astype(np.int32)
        
        class_id = res[1][0][index].cpu().detach().numpy()
        return cam.cpu().detach().numpy(), box, class_id, score.cpu().detach().numpy()

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