import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.dirname(os.path.realpath(parent_dir))
# Add the parent directory to sys.path
sys.path.append(base_dir)
##

import degirum as dg, mytools, cv2

zoo = dg.connect(dg.CLOUD, mytools.get_cloud_zoo_url(), mytools.get_token())

# Load and configure model
model = zoo.load_model('visdrone_yolov5n_relu6--640x640_float_openvino_cpu_1')

# preprocess params
model.image_backend = "opencv"
model.input_numpy_colorspace = "BGR"
model.input_letterbox_fill_color = (114, 114, 114)

# post-process params
model.overlay_show_pcdrobabilities = True # show class probabilities on overlay image
model.overlay_line_width = 1
model.overlay_alpha = 1 # set minimum transparency for overlay image labels
model.output_max_detections = 300
model.output_max_detections_per_class = 200
model.output_confidence_threshold = 0.001
model.output_nms_threshold = 0.7
model.output_max_classes_per_detection = 1
model.output_use_regular_nms = True

from ultralytics.data.augment import LetterBox
import numpy as np

path = "/data/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"

# Manual pre-process (like in ultralytics)
im0 = cv2.imread(path)  # BGR
print(im0.shape)
im = LetterBox([640, 640], auto=False, stride=32)(image=im0)
im_t = im[...,::-1]# BGR to RGB
im_flat = im_t.flatten()
index_im = np.where(im_flat != 114)

print(im.shape)


# psdk pre-process
model.input_image_format = "RAW"
pre = model._preprocessor.forward(path)[0]
print(len(pre))
pre_np = np.frombuffer(pre, np.uint8)

index_pre = np.where(pre_np != 114)

diff = im_flat - pre_np

index_diff = np.where(diff != 0)

for i in index_diff[0]:
    print(i, im_flat[i], pre_np[i])

