#
# mytools.py: toolkit for PySDK samples
# 
# Copyright DeGirum Corporation 2022
# All rights reserved
#

import degirum as dg # import DeGirum PySDK
import os, string, dotenv, cv2
from contextlib import contextmanager


# list of possible inference options
inference_option_list = {
    1: ("DeGirum Cloud Platform",                          "dgcps://cs.degirum.com", "DEGIRUM_CLOUD_TOKEN"),
    2: ("AI server connected via P2P VPN",                 "P2P_VPN_SERVER_ADDRESS", ""),
    3: ("AI server in your local network",                 "LOCAL_NETWORK_SERVER_ADDRESS", ""),
    4: ("AI server running on this machine",               "localhost", ""),
    5: ("DeGirum Orca installed on this machine",          "", "GITHUB_TOKEN")
}


def connect_model_zoo(inference_option = 1):
    """ Connect to model zoo according to given inference option """

    def _get_var(var):
        ret = os.getenv(var) if var.isupper() else var
        if ret is None:
            raise Exception(f"Please define environment variable {var} in .env file located in your CWD")
        return ret
    dotenv.load_dotenv(override=True) # load environment variables from .env file
    my_cfg = inference_option_list[inference_option]
    zoo = dg.connect_model_zoo(_get_var(my_cfg[1]), _get_var(my_cfg[2])) # connect to the model zoo
    print(f"Inference option = '{my_cfg[0]}")
    return zoo


#
# Some helper functions
#

def open_video_stream(camera_index):
    """ Open video stream from local camera """
    stream = cv2.VideoCapture(camera_index)
    if not stream.isOpened():
        raise Exception("Error opening video stream")
    else:
        print("Succesfully opened video stream")
    return stream


def show(img, capt = "<image>"):
    """ Show opencv image """
    cv2.imshow(capt, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x') or key == ord('q'):
        raise KeyboardInterrupt

    
def crop(img, bbox):
    """ Crop opencv image to given bbox """
    return img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


@contextmanager
def cv_loop():
    """ To run OpenCV actions in a loop with proper cleanup """
    try:
        yield

    except KeyboardInterrupt:
        pass # ignore KeyboardInterrupt errors
    finally:
        cv2.destroyAllWindows() # close OpenCV windows

def video_source(stream):
    """ Generator function, which returns video frames captured from given video stream
        Useful to pass to model batch_predict()
    """
    while True:
        ret, frame = stream.read()
        if not ret:
            raise Exception("Fail to capture camera frame. May be camera was opened by another notebook?")
        yield frame
