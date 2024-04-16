import degirum as dg
import numpy as np
from numpy.lib.stride_tricks import as_strided
from decoder import OpenPoseDecoder

class HumanPosePostprocessor(dg.postprocessor.DetectionResults):
    
    CATEGORY_ID = 0
    CONFIDENCE_SCORE = 1.0
    OUTPUT_SHAPE = [1, 38, 32, 57]
    POINT_SCORE_THRESHOLD = 0.1

    decoder = OpenPoseDecoder()
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        new_inference_results = []
        pafs, heatmaps = self._inference_results[0]["data"], self._inference_results[1]["data"]
        poses, scores = self._process_results(pafs, heatmaps)
        
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            landmarks = []
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > self.POINT_SCORE_THRESHOLD:
                    landmark = {
                        'category_id': i,
                        'landmark': list(p),
                        'score': v
                    }
                    landmarks.append(landmark)

            result = {
                'category_id': 0,
                'label': "human",
                'score': 1.0,
                'landmarks': landmarks
            }
            new_inference_results.append(result)

        self._inference_results = new_inference_results


    # 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)   
    def _pool2d(self, A, kernel_size, stride, padding, pool_mode="max"):
        """
        2D Pooling

        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        """
        # Padding
        A = np.pad(A, padding, mode="constant")

        # Window view of A
        output_shape = (
            (A.shape[0] - kernel_size) // stride + 1,
            (A.shape[1] - kernel_size) // stride + 1,
        )
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(
            A,
            shape=output_shape + kernel_size,
            strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
        )
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling.
        if pool_mode == "max":
            return A_w.max(axis=(1, 2)).reshape(output_shape)
        elif pool_mode == "avg":
            return A_w.mean(axis=(1, 2)).reshape(output_shape)


    # non maximum suppression
    def _heatmap_nms(self, heatmaps, pooled_heatmaps):
        return heatmaps * (heatmaps == pooled_heatmaps)


    # Get poses from results.
    def _process_results(self, pafs, heatmaps):
        # This processing comes from
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
        pooled_heatmaps = np.array(
            [[self._pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
        )
        nms_heatmaps = self._heatmap_nms(heatmaps, pooled_heatmaps)
        # Decode poses.
        poses, scores = self.decoder(heatmaps, nms_heatmaps, pafs)
        
        output_scale = self._input_image.shape[1] / self.OUTPUT_SHAPE[3], self._input_image.shape[0] / self.OUTPUT_SHAPE[2]
        # Multiply coordinates by a scaling factor.
        poses[:, :, :2] *= output_scale
        return poses, scores