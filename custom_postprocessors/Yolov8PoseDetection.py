import degirum as dg
import numpy as np
from Yolov8postprocess_utils import *
class Yolov8PoseDetection(dg.postprocessor.DetectionResults):
    
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        new_inference_results=[]
        preds = []
        labels_dict = {"LandmarkLabels": ["Nose", "LeftEye", "RightEye", "leftLipCorner", "RightLipCorner"]}
        
        for el in self._inference_results:
            prediction = el["data"]
            de_quantization_zero_parameter = el["quantization"]["zero"]
            de_quantization_scale_parameter = el["quantization"]["scale"]   
            prediction = (prediction - de_quantization_zero_parameter) * de_quantization_scale_parameter
            preds.append(prediction)

        mcv = float('-inf')
        lci = -1
        for idx, s in enumerate(preds):
            dim_1 = s.shape[1]
            if dim_1 > mcv:
                mcv = dim_1
                lci = idx
        
        pred_order = [item for index, item in enumerate(preds) if index not in [lci]]
        pred_decoded = decode_bbox(pred_order,img_shape = [1, 3, 640, 640])
        kpt_shape = (preds[lci].shape[-1] // 3, 3)     
        pred_kpts_permuted = np.transpose(preds[lci], (0, 2, 1))
        kpts_decoded = decode_kpts(pred_order, [1,3,640,640], pred_kpts_permuted, kpt_shape, bs=1)
        pred_order = np.concatenate([pred_decoded, kpts_decoded], 1)
        preds= non_max_suppression(pred_order,
                                conf_thres=0.25,
                                iou_thres = 0.8,
                                agnostic=False,
                                max_det=300,
                                classes=None,
                                nc=1)
        
        for pred in preds:
            pred_kpts = pred[:, 6:].reshape(len(pred),kpt_shape[0],kpt_shape[1])  
#             pred[:, :4] = scale_boxes([640,640], pred[:, :4], org_img_shape).round()
#             pred_kpts = scale_coords([640,640], pred_kpts,  org_img_shape)
            
            landmarks = []
            for idx, label in enumerate(labels_dict["LandmarkLabels"]):
                landmark = {
                    'label': label,
                    'category_id': idx,
                    'connect': [],
                    'landmark': pred_kpts[0][idx][:2].tolist(),
                    'score': float(pred_kpts[0][idx][2])
                }
                landmarks.append(landmark)

            result = {
                'bbox': pred[:, :4].flatten().tolist(),
                'category_id': 0,
                'label': "Human Face",
                'score': float(pred[:, 4]),
                'landmarks': landmarks
            }
            new_inference_results.append(result)              
        self._inference_results = new_inference_results
            