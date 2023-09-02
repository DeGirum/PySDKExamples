import numpy as np
import math
import time

import cv2
from PIL import Image


class DFL:
    def __init__(self, c1=16):
        """Integral module of Distribution Focal Loss (DFL)."""
        super().__init__()
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on numpy array 'x' and returns a numpy array."""
        b, c, a = x.shape  # batch, channels, anchors
        x = x.reshape((b, 4, self.c1, a))
        x = x.transpose(0, 2, 1, 3)
        x = softmax(x)
        weights = np.arange(self.c1)
        weights = np.reshape(weights, (1, self.c1, 1, 1))
        output = np.zeros((1, 1, 4, a))
        for i in range(4):
            for j in range(a):
                output[0, 0, i, j] = np.sum(x[0, :, i, j] * weights[0, :, 0, 0])

        output = output.reshape(b, 4, a)
        return output


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)  # only difference


def make_anchors(feats, strides, grid_cell_offset=0.5):
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


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, axis=1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)
    return np.concatenate((c_xy, wh), dim)  # xyxy bbox


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_bbox(preds, img_shape):
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
    dfl = DFL(reg_max) if reg_max > 1 else np.identity()
    img_h, img_w = img_shape[-2], img_shape[-1]
    strides = [
        int(math.sqrt(img_shape[-2] * img_shape[-1] / preds[p].shape[1]))
        for p in pos
        if preds[p].shape[2] != 64
    ]
    dims = [(img_h // s, img_w // s) for s in strides]
    fake_feats = [np.zeros((1, 1, h, w)) for h, w in dims]
    anchors, strides = (
        x.transpose(1, 0) for x in make_anchors(fake_feats, strides, 0.5)
    )  # generate anchors and strides
    box = x[:, :-num_classes, :]
    dbox = (
        dist2bbox(dfl.forward(box), np.expand_dims(anchors, axis=0), xywh=True, dim=1)
        * strides
    )
    cls = x[:, -num_classes:, :]
    y = np.concatenate((dbox, sigmoid(cls)), 1)
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def nms(boxes, overlap_threshold=0.2, min_mode=False):
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


def non_max_suppression(
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
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (np.ndarray): A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, numpy array]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[numpy array]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
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
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

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
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls[:, :], axis=1, keepdims=True)
            x = np.concatenate((box, conf, j, mask), axis=1)

        #         # Filter by class
        #         if classes is not None:
        #             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

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

        scores = scores.reshape(scores.shape[0], 1)
        con = np.concatenate((boxes, scores), axis=1)
        keep_boxes = nms(con, iou_thres)  # NMS
        keep_boxes = keep_boxes[:max_det]  # limit detections

        for k in keep_boxes:
            output.append(x[k])
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image in the format of (height, width).
        boxes (np.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular rescaling.

    Returns:
        boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])
    return boxes


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (np.ndarray): [n, h, w] tensor of masks
        boxes (np.ndarray): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (np.ndarray): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1))
    r = np.arange(w)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h)[None, :, None]  # cols shape(1,h,1)
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (np.ndarray): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (np.ndarray): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (np.ndarray): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag whether to upsample the mask to the original image size.
    Returns:
        (np.ndarray): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape
    ih, iw = shape

    protos = protos.astype(float)
    protos = protos.reshape(c, (protos.shape[1] * protos.shape[2]))
    n_shape = (masks_in @ protos).shape
    masks = sigmoid(masks_in @ protos).reshape(n_shape[0], mw, mh)
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih
    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    mask_ = np.where(masks > 0.5, 1, 0)
    return mask_.astype(float)


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (torch.Tensor): The masks that are being returned.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(
            im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
        )  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
            im1_shape[0] - im0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
    if len(masks.shape) < 2:
        raise ValueError(
            f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
        )
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
