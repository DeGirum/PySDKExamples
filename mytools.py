#
# mytools.py: toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2022
# All rights reserved
#

import degirum as dg  # import DeGirum PySDK
import os, time, string, dotenv, cv2, PIL, IPython.display
from contextlib import contextmanager


# list of possible inference options
inference_option_list = {
    1: ("DeGirum Cloud Platform", "dgcps://cs.degirum.com", "DEGIRUM_CLOUD_TOKEN"),
    2: ("AI server connected via P2P VPN", "P2P_VPN_SERVER_ADDRESS", ""),
    3: ("AI server in your local network", "LOCAL_NETWORK_SERVER_ADDRESS", ""),
    4: ("AI server running on this machine", "localhost", ""),
    5: ("DeGirum Orca installed on this machine", "", "GITHUB_TOKEN"),
}


def connect_model_zoo(inference_option=1):
    """Connect to model zoo according to given inference option"""

    def _get_var(var):
        ret = os.getenv(var) if var.isupper() else var
        if ret is None:
            raise Exception(
                f"Please define environment variable {var} in .env file located in your CWD"
            )
        return ret

    dotenv.load_dotenv(override=True)  # load environment variables from .env file
    my_cfg = inference_option_list[inference_option]
    zoo = dg.connect_model_zoo(
        _get_var(my_cfg[1]), _get_var(my_cfg[2])
    )  # connect to the model zoo
    print(f"Inference option = '{my_cfg[0]}'")
    return zoo


#
# Some helper functions
#


def open_video_stream(camera_id=None):
    """Open video stream from camera with given identifier
    camera_id - 0-based index for local cameras
       or IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>"
    """
    if camera_id is None:
        dotenv.load_dotenv(override=True)  # load environment variables from .env file
        camera_id = os.getenv("CAMERA_ID")
        if camera_id.isnumeric():
            camera_id = int(camera_id)
    if camera_id is None:
        raise Exception(
            "No camera ID specified. Either define 'CAMERA_ID' environment variable or pass as a parameter"
        )
    stream = cv2.VideoCapture(camera_id)
    if not stream.isOpened():
        raise Exception(f"Error opening '{camera_id}' video stream")
    else:
        print(f"Successfully opened video stream '{camera_id}'")
    return stream


class FPSMeter:
    """Simple FPS meter class"""

    def __init__(self, avg_len=100):
        """Constructor
        avg_len - number of samples to average
        """
        self._avg_len = avg_len
        self.reset()

    def reset(self):
        """Reset accumulators"""
        self._timestamp_ns = -1
        self._duration_ns = -1
        self._count = 0

    def record(self):
        """Record timestamp and update average duration
        Return current average FPS"""
        t = time.time_ns()
        if self._timestamp_ns > 0:
            cur_dur_ns = t - self._timestamp_ns
            self._count = min(self._count + 1, self._avg_len)
            self._duration_ns = (
                self._duration_ns * (self._count - 1) + cur_dur_ns
            ) / self._count
        self._timestamp_ns = t
        return self.fps()

    def fps(self):
        """Return current average FPS"""
        return 1e9 / self._duration_ns if self._duration_ns > 0 else 0


class Display:
    """Class to handle OpenCV image display"""

    def __init__(self, capt="<image>", show_fps=True, show_embedded=False):
        """Constructor
        show_fps - True to show FPS
        capt - window title
        """
        self._fps = FPSMeter() if show_fps else None
        self._capt = capt
        self._need_destroy = False
        self._show_embedded = show_embedded
        self._no_gui = not Display._check_gui()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._need_destroy:
            cv2.destroyWindow(self._capt)  # close OpenCV window
        return exc_type is KeyboardInterrupt  # ignore KeyboardInterrupt errors

    def crop(img, bbox):
        """Crop opencv image to given bbox"""
        return img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

    def _check_gui():
        """Check if graphical display is supported"""
        import os, platform

        if platform.system() == "Linux":
            return os.environ.get("DISPLAY") is not None
        return True

    def _show_fps(img, fps):
        """Helper method to display FPS"""
        text = f"{fps:5.1f} FPS"
        font = cv2.FONT_HERSHEY_PLAIN
        text_size = cv2.getTextSize(text, font, 1, 1)
        text_w = text_size[0][0]
        text_h = text_size[0][1] + text_size[1]
        margin = int(text_h / 4)
        bl_corner = (margin, img.shape[0] - margin)
        tr_corner = (
            bl_corner[0] + text_w + margin,
            bl_corner[1] - text_h - margin,
        )
        cv2.rectangle(img, bl_corner, tr_corner, (255, 255, 255), cv2.FILLED)
        cv2.putText(
            img,
            text,
            (bl_corner[0] + margin, bl_corner[1] - margin),
            font,
            1,
            (0, 0, 0),
        )

    def show(self, img):
        """Show OpenCV image
        img - numpy array with valid OpenCV image
        """
        if self._fps:
            fps = self._fps.record()
            if fps > 0:
                Display._show_fps(img, fps)

        if self._show_embedded or self._no_gui:
            IPython.display.display(PIL.Image.fromarray(img[..., ::-1]), clear=True)
        else:
            cv2.imshow(self._capt, img)
            self._need_destroy = True
            key = cv2.waitKey(1) & 0xFF
            if key == ord("x") or key == ord("q"):
                if self._fps:
                    self._fps.reset()
                raise KeyboardInterrupt


def video_source(stream):
    """Generator function, which returns video frames captured from given video stream
    Useful to pass to model batch_predict()
    """
    while True:
        ret, frame = stream.read()
        if not ret:
            raise Exception(
                "Fail to capture camera frame. May be camera was opened by another notebook?"
            )
        yield frame
