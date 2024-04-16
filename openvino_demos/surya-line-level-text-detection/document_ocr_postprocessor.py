from degirum import postprocessor
import numpy as np
import cv2
from heatmap import get_and_clean_boxes

class DocTextDetPostprocessor(postprocessor.DetectionResults):
    PREDICTION_OUTPUT_SHAPE = (2,224,224)
    TEXT_CATEGORY_ID = 0
    CONFIDENCE_SCORE = 1.0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        reshaped_prediction = self._inference_results[0]["data"].reshape((1,) + self.PREDICTION_OUTPUT_SHAPE)
        resized_heatmap = self._generate_and_resize_heatmap(reshaped_prediction)
        results = self._get_boxes(resized_heatmap)

        self._inference_results = [
            self._create_result_dict(bbox)
            for bbox in results[0]["bboxes"]
        ]

    def _generate_and_resize_heatmap(self, reshaped_prediction):
        heatmap = reshaped_prediction[0, 0, :, :].astype(np.float32)
        heatmap_shape = list(heatmap.shape)
        correct_shape = [self._model_params.InputW[0],self._model_params.InputH[0]]
        cv2_size = list(reversed(correct_shape)) # opencv uses (width, height) instead of (height, width)
        if heatmap_shape != correct_shape:
            heatmap = cv2.resize(heatmap, cv2_size, interpolation=cv2.INTER_LINEAR)
        return heatmap
    
    def _get_boxes(self, resized_heatmap):
        results = []
        orig_sizes = [self._input_image.size]
        heatmap_size = list(reversed(resized_heatmap.shape))
        bboxes = get_and_clean_boxes(resized_heatmap, heatmap_size, orig_sizes[0])
        bbox_data = [bbox.model_dump() for bbox in bboxes]
        results.append({
            "bboxes": [bbd["bbox"] for bbd in bbox_data]
        })
        return results
    
    def _create_result_dict(self, bbox):
        return {
            "bbox": bbox,
            "category_id": self.TEXT_CATEGORY_ID,
            "label": "text",
            "score": self.CONFIDENCE_SCORE,
        }
    
    