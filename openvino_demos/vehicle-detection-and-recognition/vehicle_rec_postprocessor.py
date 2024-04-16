import degirum as dg
import numpy as np

class VehicleRecPostprocessor(dg.postprocessor.ClassificationResults):
    COLORS = ['White', 'Gray', 'Yellow', 'Red', 'Green', 'Blue', 'Black']
    TYPES = ['Car', 'Bus', 'Truck', 'Van']
    VEHICLE_CATEGORY_ID = 0
    CONFIDENCE_SCORE = 1.0

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        predict_types, predict_colors = self._inference_results[0]["data"],self._inference_results[1]["data"]
        # print (predict_types, predict_colors)
        predict_colors = np.squeeze(predict_colors, (2, 3))
        predict_types = np.squeeze(predict_types, (2, 3))

        attr_color, attr_type = (self.COLORS[np.argmax(predict_colors)],
                                self.TYPES[np.argmax(predict_types)])
        # print (attr_color, attr_type)
        result = {
            "label": attr_color+" "+attr_type,
            "category_id": self.VEHICLE_CATEGORY_ID,
            "score": self.CONFIDENCE_SCORE,
        }
        self._inference_results = [result]