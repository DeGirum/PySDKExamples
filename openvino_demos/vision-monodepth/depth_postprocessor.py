import importlib
import importlib.util

import numpy as np
import degirum as dg
from matplotlib import colormaps

class DepthResults(dg.postprocessor.InferenceResults):

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
            _resize_map = {'nearest': self._backend.INTER_NEAREST,
                           'bilinear': self._backend.INTER_LINEAR,
                           'area': self._backend.INTER_AREA,
                           'bicubic': self._backend.INTER_CUBIC,
                           'lanczos': self._backend.INTER_LANCZOS4}

        elif importlib.util.find_spec("PIL"):
            self._backend = importlib.import_module("PIL")
            if self._backend and isinstance(self.image, self._backend.Image.Image):
                self._backend_name  = 'pil'
                _resize_map = {'nearest': self._backend.Image.Resampling.NEAREST,
                               'bilinear': self._backend.Image.Resampling.BILINEAR,
                               'area': self._backend.Image.Resampling.BOX,
                               'bicubic': self._backend.Image.Resampling.BICUBIC,
                               'lanczos': self._backend.Image.Resampling.LANCZOS}

        # Resize the depth map back to the original image size.
        data = self._inference_results[0]['data']
        data = np.transpose(data, (1, 2, 0))
        resize_mode = _resize_map[self._model_params.InputResizeMethod[0]]

        if self._backend_name == 'cv2':
            image_size = self.image.shape[:2][::-1]
            data = self._backend.resize(data, image_size, interpolation=resize_mode)
        else:
            data_surrogate = self._backend.Image.fromarray(data.squeeze())
            data_surrogate = data_surrogate.resize(self.image.size, resample=resize_mode)
            data = np.array(data_surrogate)

        data = np.expand_dims(data, axis=0)
        self._inference_results[0]['data'] = data  # TODO: Should the returned depth map be normalized already?

    def _normalize_depth_map(self, depth_map):
        return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    def _convert_depth_to_image(self, depth_map, color_map='viridis'):
        c_map = colormaps[color_map]
        depth_map = depth_map.squeeze(0)
        depth_map = self._normalize_depth_map(depth_map)
        depth_map = c_map(depth_map)[:, :, :3] * 255
        depth_map = depth_map.astype(np.uint8)

        return depth_map
    
    @property
    def image_overlay(self):
        image = self._convert_depth_to_image(self._inference_results[0]['data'])

        if self._backend_name == 'cv2':
            return image
        
        return self._backend.Image.fromarray(image)
    
    def __repr__(self):
        return self._inference_results.__repr__()
    
    def __str__(self):
        return self._inference_results.__str__()
