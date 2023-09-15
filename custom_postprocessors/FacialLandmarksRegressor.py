import degirum as dg

class FacialLandmarkRegressor(dg.postprocessor.DetectionResults):   
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        new_inference_results=[]
        pred_order = []
        for el in self._inference_results:
            prediction = el["data"]
            faceheight, facewidth, channels = self._input_image.shape
            left_eye_x, left_eye_y = int(prediction[0][0][0][0] * facewidth), int(prediction[0][1][0][0] * faceheight)
            right_eye_x, right_eye_y = int(prediction[0][2][0][0] * facewidth), int(prediction[0][3][0][0] * faceheight)
            nose_x, nose_y = int(prediction[0][4][0][0] * facewidth), int(prediction[0][5][0][0] * faceheight)
            lip_left_corner_x, lip_left_corner_y = int(prediction[0][6][0][0] * facewidth), int(prediction[0][7][0][0] * faceheight)
            lip_right_corner_x, lip_right_corner_y = int(prediction[0][8][0][0] * facewidth), int(prediction[0][9][0][0] * faceheight)

            keypoints = [left_eye_x,left_eye_y,right_eye_x,right_eye_y,nose_x,nose_y,lip_left_corner_x,lip_left_corner_y,lip_right_corner_x,lip_right_corner_y]
            
            result = {"landmarks" : [{"label" : 'LeftEye', "category_id" : 0, "landmark":keypoints[0:2]},{"label" : 'RightEye', "category_id" : 1, "landmark":keypoints[2:4]},{"label":'Nose',"category_id" : 2,"landmark":keypoints[4:6]},{"label":'LipsleftCorner',"category_id" : 3,"landmark":keypoints[6:8]},{"label":'LipsRightCorner',"category_id" : 4,"landmark":keypoints[8:10]}]}
            new_inference_results.append(result)
            
        self._inference_results = new_inference_results