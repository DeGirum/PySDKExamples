import importlib
import importlib.util

import numpy as np
import degirum as dg
from matplotlib import colormaps

import cv2
from PIL.Image import Image as PILImage

def preprocess(image, model):
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
    elif isinstance(image, PILImage):
        w, h = image.size
        image = np.array(image)
    else:
        raise Exception("Image type not supported.")
    
    padded_image = np.zeros((model.model_info.InputH[0], model.model_info.InputW[0], model.model_info.InputC[0]), dtype=np.uint8)
    padded_image[:h, :w] = image

    return padded_image

class SuperResolutionResults(dg.postprocessor.InferenceResults):
    resize_factor = 4
    dynamic_input = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # call base class constructor first

        # Retrieve backend
        if (
            isinstance(self.image, np.ndarray)
            and len(self.image.shape) == 3
            and importlib.util.find_spec("cv2")
        ):
            self._backend = importlib.import_module("cv2")
            self._backend_name = 'cv2'

        elif importlib.util.find_spec("PIL"):
            self._backend = importlib.import_module("PIL")
            if self._backend and isinstance(self.image, self._backend.Image.Image):
                self._backend_name  = 'pil'

        # Normalize and remove padding
        data = self._inference_results[0]['data']
        data = (data.squeeze() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0) # Currently in RGB
        nh, nw, _ = data.shape

        if self._backend_name == 'cv2':
            h, w, _ = self.image.shape
        else:
            w, h = self.image.size

        if SuperResolutionResults.dynamic_input:  # This is when we pad for dynamic input type models

            # sh = h / self._model_params.InputH[0]
            # sw = w / self._model_params.InputW[0]

            # if sh > 0:

            data = data[:h * SuperResolutionResults.resize_factor, :w * SuperResolutionResults.resize_factor]
        else:  # This is for fixed size models, INTER_AREA for downsizing, INTER_LINEAR for upsizing.
            if nh >= h * SuperResolutionResults.resize_factor or nw >= w * SuperResolutionResults.resize_factor:
                resize_method = cv2.INTER_AREA
            else:
                resize_method = cv2.INTER_LINEAR

            data = cv2.resize(data, (w * SuperResolutionResults.resize_factor, h * SuperResolutionResults.resize_factor), resize_method)

    
        self._inference_results[0]['data'] = data  
    
    @property
    def image_overlay(self):
        if self._backend_name == 'cv2':
            return self._backend.cvtColor(self._inference_results[0]['data'], self._backend.COLOR_RGB2BGR)
        
        return self._backend.Image.fromarray(self._inference_results[0]['data'])
    
    def __repr__(self):
        return self._inference_results.__repr__()
    
    def __str__(self):
        return self._inference_results.__str__()