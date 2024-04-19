
import degirum as dg

class OCRTextDetPostprocessor(dg.postprocessor.DetectionResults):
    PREDICTION_ELEMENTS = 5
    TEXT_CATEGORY_ID = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        reshaped_predictions = self._inference_results[0]["data"].reshape(
            (-1, self.PREDICTION_ELEMENTS)
        )
        self._inference_results = [
            self._create_result_dict(prediction) for prediction in reshaped_predictions
        ]

    def _create_result_dict(self, prediction):
        bbox, score = prediction[:4], prediction[-1]
        bbox = [*self._conversion(*bbox[:2]), *self._conversion(*bbox[2:])]
        return {
            "bbox": bbox,
            "category_id": self.TEXT_CATEGORY_ID,
            "label": "text",
            "score": score,
        }