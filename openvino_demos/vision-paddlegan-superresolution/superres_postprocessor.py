import importlib
import importlib.util
import warnings

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
    
    # The entire preprocessing is a workaround until PySDK supports dynamic inputs.
    if model.model_info.InputType[0] == "Tensor":
        model_h, model_w = model.model_info.InputW[0], model.model_info.InputC[0]
    else:
        model_h, model_w = model.model_info.InputH[0], model.model_info.InputW[0]

    if h > model_h or w > model_w:
        warnings.warn("Image larger than model size. Inference will still run but expect some loss in detail.")
        # Do semi-letterbox.
        if h > w:
            nh = model_h
            nw = int(w / h * nh)
        else:
            nw = model_w
            nh = int(w / h * nw)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        nw, nh = w, h
        
    padded_image = np.zeros((model_h, model_w, 3), dtype=np.uint8)
    padded_image[:nh, :nw] = image

    if model.custom_postprocessor == None:
        raise Exception("Set custom super resolution postprocessor before preprocessing images.")
    
    model.custom_postprocessor.dynamic_input = True
    model.custom_postprocessor.params = (nw, nh, w, h)
    return padded_image

class SuperResolutionResults(dg.postprocessor.InferenceResults):
    resize_factor = 4
    dynamic_input = False
    params = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # call base class constructor first

        # Retrieve backend
        if self.image is None:
            # Tensor input type default to cv2.
            self._backend = importlib.import_module("cv2")
            self._backend_name = 'cv2'
        else:
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

        r_factor = SuperResolutionResults.resize_factor

        # Normalize and remove padding
        data = self._inference_results[0]['data']
        if len(data.shape) != 4:
            if self._model_params.InputType[0] == "Image":
                data = np.reshape(data, (1, 3, self._model_params.InputH[0] * r_factor, self._model_params.InputW[0] * r_factor))
            else:
                data = np.reshape(data, (1, 3, self._model_params.InputW[0] * r_factor, self._model_params.InputC[0] * r_factor))
        data = (data.squeeze() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0) # Currently in RGB
        nh, nw, _ = data.shape

        try:
            if self._backend_name == 'cv2':
                h, w, _ = self.image.shape
            else:
                w, h = self.image.size
        except Exception as e:
            pass

        # This is the workaround for dynamic input until PySDK supports dynamic input.
        if SuperResolutionResults.dynamic_input:
            dnw, dnh, dw, dh = SuperResolutionResults.params
            data = data[:dnh * r_factor, :dnw * r_factor]
            if dw != dnw or dh != dnh:
                data = cv2.resize(data, (dw * r_factor, dh * r_factor), interpolation=cv2.INTER_CUBIC)
        else:  # This is for fixed size models, INTER_AREA for downsizing, INTER_LINEAR for upsizing.
            if nh >= h * r_factor or nw >= w * r_factor:
                resize_method = cv2.INTER_AREA
            else:
                resize_method = cv2.INTER_CUBIC

            data = cv2.resize(data, (w * r_factor, h * r_factor), resize_method)

    
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