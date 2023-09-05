import degirum as dg
import numpy as np
from postprocess_utils import *
class Yolov8Segmentation(dg.postprocessor.DetectionResults):
    
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        new_inference_results=[]
        pred_order = []
        for el in self._inference_results:
            prediction = el["data"]
            pred_order.append(prediction)

        preds_decoded = decode_bbox(pred_order[:6], [1, 3, 640, 640])
        mask = pred_order[6]
        proto = pred_order[7]
        proto = proto.transpose(0,3,1,2)
        nc = preds_decoded.shape[1] - 4
        preds_decoded = np.concatenate([preds_decoded, np.transpose(mask,(0, 2, 1))], 1)

        p = non_max_suppression(preds_decoded,conf_thres=0.25, iou_thres=0.7, classes=None, agnostic=False, multi_label=False, max_det=300, nc=nc)
        pred = np.vstack(p)
        masks = process_mask(proto[0], pred[:, 6:], pred[:, :4],(640,640))  # HWC
        masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
        
        # Rescale masks to original image
        masks = scale_image(masks,self._input_image.shape)
        masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W))
        for p in pred[:, :6]:
            result = {"bbox" : p[:4].tolist(), "score" : p[4] , "category_id" : int(p[5]), "masks" : masks}
            box = result["bbox"]
            result["bbox"] = [
                *self._conversion(*box[:2]),
                *self._conversion(*box[2:]),
            ]
            
            new_inference_results.append(result)
            
        self._inference_results = new_inference_results
            