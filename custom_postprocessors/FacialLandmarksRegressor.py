import degirum as dg

class FacialLandmarkRegressor(dg.postprocessor.DetectionResults):   
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        new_inference_results=[]
        pred_order = []
        for el in self._inference_results:
            prediction = el["data"]
            faceheight, facewidth, channels = self._input_image.shape
            # labels for facial keypoints
            keypoint_labels = ['LeftEye', 'RightEye', 'Nose', 'LipsleftCorner', 'LipsRightCorner']
            keypoints = []
            # Iterate over the 5 keypoints
            for i in range(5):
                x = int(prediction[0][i * 2][0][0] * facewidth) # Extract the x-coordinate of the keypoint and scale it by the face width
                y = int(prediction[0][i * 2 + 1][0][0] * faceheight) # Extract the y-coordinate of the keypoint and scale it by the face height
                keypoints.extend([x, y])
            result = {"landmarks": [{"label": label, "category_id": idx, "landmark": keypoints[idx * 2:idx * 2 + 2]} for idx, label in enumerate(keypoint_labels)]}
            new_inference_results.append(result)
            
        self._inference_results = new_inference_results
