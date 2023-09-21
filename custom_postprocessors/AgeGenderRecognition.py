import degirum as dg

class AgeGenderRecognition(dg.postprocessor.DetectionResults):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        new_inference_results=[]
        
        gender_prediction = self._inference_results[0]["data"]
        age_prediction = self._inference_results[1]["data"]
        
        if gender_prediction[0][0][0][0] > gender_prediction[0][1][0][0]:
            result = {"gender": "Female","age" : int(age_prediction[0][0][0][0] * 100)}
        else:
            result = {"gender": "Male","age" : int(age_prediction[0][0][0][0] * 100)}
            
        new_inference_results.append(result)
        self._inference_results = new_inference_results