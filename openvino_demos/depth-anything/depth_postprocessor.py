import importlib
import importlib.util
import numpy as np
import degirum as dg
from matplotlib import pyplot as plt

class DepthResults(dg.postprocessor.InferenceResults):
    """Class to process depth inference results."""

    color_map = "viridis"

    def __init__(self, *args, **kwargs):
        """Initialize the DepthResults object."""
        super().__init__(*args, **kwargs)  # call base class constructor first

        # Retrieve backend
        self._backend, self._backend_name, _resize_map = self._get_backend_and_resize_map()

        # Resize the depth map back to the original image size if the original image exists.
        if self.image is not None:
            self._resize_depth_map(_resize_map)

    def _get_backend_and_resize_map(self):
        """Retrieve the appropriate backend and resize map based on the image type."""
        if (
            isinstance(self.image, np.ndarray)
            and len(self.image.shape) == 3
            and importlib.util.find_spec("cv2")
        ):
            backend = importlib.import_module("cv2")
            backend_name = 'cv2'
            resize_map = {'nearest': backend.INTER_NEAREST,
                          'bilinear': backend.INTER_LINEAR,
                          'area': backend.INTER_AREA,
                          'bicubic': backend.INTER_CUBIC,
                          'lanczos': backend.INTER_LANCZOS4}
        elif importlib.util.find_spec("PIL"):
            backend = importlib.import_module("PIL")
            if backend and isinstance(self.image, backend.Image.Image):
                backend_name = 'pil'
                resize_map = {'nearest': backend.Image.Resampling.NEAREST,
                              'bilinear': backend.Image.Resampling.BILINEAR,
                              'area': backend.Image.Resampling.BOX,
                              'bicubic': backend.Image.Resampling.BICUBIC,
                              'lanczos': backend.Image.Resampling.LANCZOS}
        else:
            raise RuntimeError("No suitable backend found for image processing.")

        return backend, backend_name, resize_map

    def _resize_depth_map(self, resize_map):
        """Resize the depth map to match the original image size."""
        data = self._inference_results[0]['data']
        data = np.transpose(data, (1, 2, 0))
        resize_mode = resize_map[self._model_params.InputResizeMethod[0]]

        if self._backend_name == 'cv2':
            image_size = self.image.shape[:2][::-1]
            data = self._backend.resize(data, image_size, interpolation=resize_mode)
        else:
            data_surrogate = self._backend.Image.fromarray(data.squeeze())
            data_surrogate = data_surrogate.resize(self.image.size, resample=resize_mode)
            data = np.array(data_surrogate)

        data = np.expand_dims(data, axis=0)
        self._inference_results[0]['data'] = data

    def _normalize_depth_map(self, depth_map):
        """Normalize the depth map for better visualization."""
        return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    def _convert_depth_to_image(self, depth_map):
        """Convert the depth map to an image using a colormap."""
        c_map = plt.get_cmap(DepthResults.color_map)
        depth_map = depth_map.squeeze(0)
        depth_map = self._normalize_depth_map(depth_map)
        depth_map = c_map(depth_map)[:, :, :3] * 255
        depth_map = depth_map.astype(np.uint8)

        return depth_map

    @property
    def image_overlay(self):
        """Overlay the depth map on the original image."""
        image = self._convert_depth_to_image(self._inference_results[0]['data'])

        if self._backend_name == 'cv2':
            return image

        return self._backend.Image.fromarray(image)

    def __repr__(self):
        """Return a string representation of the object."""
        return self._inference_results.__repr__()

    def __str__(self):
        """Return a string description of the object."""
        return self._inference_results.__str__()