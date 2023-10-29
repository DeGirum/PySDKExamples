## @package Trivial Post Processor for unit tests
import torch
import torchvision
import numpy as np
import json
import math
import time
## Post-processor class
# It must have fixed name 'PostProcessor'

class PostProcessor:
    def __init__(self, json_config):
        with open(json_config, 'r') as j:
            self._json_config = json.loads(j.read())
        # self._json_config = json.loads(json_config)
        self._output_conf_threshold = float(self._json_config["POST_PROCESS"][0]["OutputConfThreshold"])
        self._output_nms_threshold = float(self._json_config["POST_PROCESS"][0]["OutputNMSThreshold"])
        self._maximum_detections_per_class = int(self._json_config["POST_PROCESS"][0]["MaxDetectionsPerClass"])
        self._input_w = int (self._json_config["PRE_PROCESS"][0]["InputW"])
        self._input_h = int(self._json_config["PRE_PROCESS"][0]["InputH"])
        self._input_c = int(self._json_config["PRE_PROCESS"][0]["InputC"])
        self._landmark_labels = self._json_config["POST_PROCESS"][0]["LandmarkLabels"]
        self._connections = self._json_config["POST_PROCESS"][0]["Connections"]
        self._label_json_config = 'yolov8n-pose/labels_coco_pose.json' # self._json_config["POST_PROCESS"][0]["LabelsPath"]
        with open(self._label_json_config, 'r') as json_file:
            labels = json.load(json_file)
            self._label = labels["0"]

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_arr = [], []
        assert feats is not None
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = np.arange(0, w, dtype=float) + grid_cell_offset
            sy = np.arange(0, h, dtype=float) + grid_cell_offset
            sy, sx = np.meshgrid(sy, sx)
            anchor_points.append(np.stack((sy, sx), axis=-1).reshape((-1, 2)))
            stride_arr.append(np.full((h * w, 1), stride, dtype=float))
        return np.concatenate(anchor_points), np.concatenate(stride_arr)


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1)  # only difference
    
    def dfl_forward(self, x, c1=16):
        """Applies a transformer layer on numpy array 'x' and returns a numpy array."""
        
        b, c, a = x.shape  # batch, channels, anchors
        x = x.reshape((b, 4, c1, a))
        x = x.transpose(0, 2, 1, 3)
        x = self.softmax(x)
        x = x.reshape((c1,-1))
        weights = np.arange(c1)
        weights = np.reshape(weights, (1, c1, 1, 1))
        weights = weights.reshape((c1,-1))
        output_1 = weights.T @ x
        output_1 = output_1.reshape(b, 4, a)
        return output_1

    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = np.split(distance, 2, axis=1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return np.concatenate((c_xy, wh), dim)
        return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def decode_bbox(self, preds, img_shape):
        """A list of predictions are decoded to shape [1, 84, 8400]"""
        num_classes = next((o.shape[2] for o in preds if o.shape[2] != 64), -1)
        assert (
            num_classes != -1
        ), "cannot infer postprocessor inputs via output shape if there are 64 classes"
        pos = [
            i
            for i, _ in sorted(
                enumerate(preds),
                key=lambda x: (
                    x[1].shape[2] if num_classes > 64 else -x[1].shape[2],
                    -x[1].shape[1],
                ),
            )
        ]
        x = np.concatenate(
            [
                np.concatenate([preds[i] for i in pos[: len(pos) // 2]], 1),
                np.concatenate([preds[i] for i in pos[len(pos) // 2 :]], 1),
            ],
            2,
        )
        x = np.transpose(x, (0, 2, 1))
        reg_max = (x.shape[1] - num_classes) // 4
        # dfl = self.dfl_forward(x, reg_max) if reg_max > 1 else np.identity()
        img_h, img_w = img_shape[-2], img_shape[-1]
        strides = [
            int(math.sqrt(img_shape[-2] * img_shape[-1] / preds[p].shape[1]))
            for p in pos
            if preds[p].shape[2] != 64
        ]
        dims = [(img_h // s, img_w // s) for s in strides]
        fake_feats = [np.zeros((1, 1, h, w)) for h, w in dims]
        anchors, strides = (
            x.transpose(1, 0) for x in self.make_anchors(fake_feats, strides, 0.5)
        )  # generate anchors and strides
        box = x[:, :-num_classes, :]
        dbox = (
            self.dist2bbox(self.dfl_forward(box, reg_max), np.expand_dims(anchors, axis=0), xywh=True, dim=1)
            * strides
        )
        cls = x[:, -num_classes:, :]
        y = np.concatenate((dbox, self.sigmoid(cls)), 1)
        return y


    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
            y (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = np.copy(x)
        y[..., 0:2] = x[..., 0:2] - x[..., 2:4] / 2  # top left (x, y)
        y[..., 2:4] = x[..., 0:2] + x[..., 2:4] / 2  # bottom right (x, y)
        return y


    def nms(self, boxes, overlap_threshold=0.2, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        index_array = scores.argsort()[::-1]
        keep = []
        while index_array.size > 0:
            keep.append(index_array[0])
            x1_ = np.maximum(x1[index_array[0]], x1[index_array[1:]])
            y1_ = np.maximum(y1[index_array[0]], y1[index_array[1:]])
            x2_ = np.minimum(x2[index_array[0]], x2[index_array[1:]])
            y2_ = np.minimum(y2[index_array[0]], y2[index_array[1:]])

            w = np.maximum(0.0, x2_ - x1_ + 1)
            h = np.maximum(0.0, y2_ - y1_ + 1)
            inter = w * h

            if min_mode:
                overlap = inter / np.minimum(areas[index_array[0]], areas[index_array[1:]])
            else:
                overlap = inter / (areas[index_array[0]] + areas[index_array[1:]] - inter)

            inds = np.where(overlap <= overlap_threshold)[0]
            index_array = index_array[inds + 1]
        return keep


    def non_max_suppression(self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.8,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=80,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
    ):
        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(
            prediction, (list, tuple)
        ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        output = []
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = np.transpose(prediction, (0, 2, 1))
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy

        t = time.time()
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = np.concatenate((x, v), axis=0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            #         # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x[:, :4], x[:, 4 : nc + 4], x[:, nc + 4 :]
            if multi_label:
                i, j = np.where(cls > conf_thres)
                x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), axis=1)
            else:  # best class only
                conf = np.max(cls, axis=1, keepdims=True)
                j = np.argmax(cls[:, :], axis=1, keepdims=True)
                x = np.concatenate((box, conf, j, mask), axis=1)

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == np.any(classes, axis=1))]

            #       # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[
                    x[:, 4].argsort(descending=True)[:max_nms]
                ]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            boxes_t = torch.from_numpy(boxes)
            scores_t = torch.from_numpy(scores)
            keep_boxes_t = torchvision.ops.nms(boxes_t, scores_t, iou_thres)
            keep_boxes = keep_boxes_t.numpy().tolist()

            # scores = scores.reshape(scores.shape[0], 1)
            # con = np.concatenate((boxes, scores), axis=1)
            # keep_boxes = self.nms(con, iou_thres)  # NMS
            
            keep_boxes = keep_boxes[:max_det]  # limit detections

            for k in keep_boxes:
                output.append(np.array([x[k]]))
        return output

def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

            
    def decode_kpts(self, preds, img_shape, kpts, kpt_shape, bs=1):
        """Decodes keypoints."""
        num_classes = next((o.shape[2] for o in preds if o.shape[2] != 64), -1)
        assert (
            num_classes != -1
        ), "cannot infer postprocessor inputs via output shape if there are 64 classes"
        pos =  [i for i,_ in sorted(enumerate(preds), key = lambda x: (x[1].shape[2] if num_classes > 64 else -x[1].shape[2], -x[1].shape[1]))]
        img_h, img_w = img_shape[-2], img_shape[-1]
        strides = [int(math.sqrt(img_shape[-2] * img_shape[-1] / preds[p].shape[1])) for p in pos if preds[p].shape[2] != 64]
        dims = [(img_h // s, img_w // s) for s in strides]
        fake_feats = [np.zeros((1, 1, h, w)) for h, w in dims]
        anchors, strides = (
            x.transpose(1, 0) for x in self.make_anchors(fake_feats, strides, 0.5))
        ndim = kpt_shape[1]
        y = kpts.copy()
        if ndim == 3:
            y[:, 2::3] = 1 / (1 + np.exp(-y[:, 2::3]))  # inplace sigmoid
            
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchors[0] - 0.5)) * strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (anchors[1] - 0.5)) * strides
        return y

    def forward(self, tensor_list, details_list):
        new_inference_results=[]
        preds_list = tensor_list
        mcv = float('-inf')
        lci = -1
        for idx, s in enumerate(preds_list):
            dim_1 = s.shape[1]
            if dim_1 > mcv:
                mcv = dim_1
                lci = idx
                
        pred_order = [item for index, item in enumerate(preds_list) if index not in [lci]]
        pred_decoded = self.decode_bbox(pred_order,img_shape =  [1, self._input_c, self._input_w, self._input_h])
        kpt_shape = (preds_list[lci].shape[-1] // 3, 3)     
        pred_kpts_permuted = np.transpose(preds_list[lci], (0, 2, 1))
        kpts_decoded = self.decode_kpts(pred_order, [1, self._input_c, self._input_w, self._input_h], pred_kpts_permuted, kpt_shape, bs=1)
        pred_order = np.concatenate([pred_decoded, kpts_decoded], 1)
        preds = self.non_max_suppression(pred_order,conf_thres=self._output_conf_threshold,iou_thres = self._output_nms_threshold,agnostic=False,max_det=self._maximum_detections_per_class,classes=None,nc=1)
        for pred in preds:
            pred_kpts = pred[:, 6:].reshape(len(pred),kpt_shape[0],kpt_shape[1])           
            landmarks = []
            for idx, label in enumerate(self._landmark_labels):
                landmark = {
                    'label': label,
                    'category_id': idx,
                    'connect': self._connections.get(str(idx),[]),
                    'landmark': pred_kpts[0][idx][:2].tolist(),
                    'score': float(pred_kpts[0][idx][2])
                }
                landmarks.append(landmark)

            result = {
                'bbox': pred[:, :4].flatten().tolist(),
                'category_id': 0,
                'label': self._label,
                'score': float(pred[:, 4]),
                'landmarks': landmarks
            }
            new_inference_results.append(result)
        return new_inference_results