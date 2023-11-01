
from benchmark_scripts.landmarks_detection_renamed import PostProcessor
import degirum as dg

PP = PostProcessor("/home/mehrdad/wa/DG/PySDKExamples/yolov8n-pose/yolov8n-pose.json")

class Yolov8PoseDetection(dg.postprocessor.DetectionResults):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        preds = []
        
        for el in self._inference_results:
            prediction = el["data"]
            de_quantization_zero_parameter = el["quantization"]["zero"]
            de_quantization_scale_parameter = el["quantization"]["scale"]   
            prediction = (prediction - de_quantization_zero_parameter) * de_quantization_scale_parameter
            preds.append(prediction)

        self._inference_results = PP.forward(preds, None)
