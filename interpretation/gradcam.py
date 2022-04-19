import numpy as np
import torch

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
        print("feature shape:{}".format(output.size()))

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
        res = self.net.bbox_head.simple_test(
            feat, data['img_metas'][0], rescale=True)
        
        score = res[0][0][index][4]
       
        score.backward()

        gradient = self.gradient[0] # [C,H,W]
        weight = torch.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0]  # [C,H,W]
        
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)  # ReLU

        # Normalization
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        # resize to 224*224
        box = res[0][0][index][:-1].cpu().detach().numpy().astype(np.int32)
        
        class_id = res[0][1][index].cpu().detach().numpy()
        return cam.cpu().detach().numpy(), box, class_id