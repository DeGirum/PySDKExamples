#
# mytools.py: toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2022
# All rights reserved
#


import os, time, string, dotenv, cv2, PIL, IPython.display
from contextlib import contextmanager


# list of possible inference options
inference_option_list = {
    1: {
        "desc": "DeGirum Cloud Platform",
        "url": "DEGIRUM_CLOUD_SERVER_ADDRESS",
        "url_default": "dgcps://cs.degirum.com",
        "token": "DEGIRUM_CLOUD_TOKEN",
    },
    2: {
        "desc": "AI server connected via P2P VPN",
        "url": "P2P_VPN_SERVER_ADDRESS",
        "url_default": None,
        "token": "",
    },
    3: {
        "desc": "AI server in your local network",
        "url": "LOCAL_NETWORK_SERVER_ADDRESS",
        "url_default": None,
        "token": "",
    },
    4: {
        "desc": "AI server running on this machine",
        "url": "127.0.0.1",
        "url_default": None,
        "token": "",
    },
    5: {
        "desc": "DeGirum Orca installed on this machine",
        "url": None,
        "url_default": None,
        "token": "GITHUB_TOKEN",
    },
}


def connect_model_zoo(inference_option=1):
    """Connect to model zoo according to given inference option

    Returns model zoo accessor object
    """

    import degirum as dg  # import DeGirum PySDK

    def _get_var(var, default_val=None):
        if var is not None and var.isupper():  # treat `var` as env. var. name
            ret = os.getenv(var)
            if ret is None:
                if default_val is None:
                    raise Exception(
                        f"Please define environment variable {var} in .env file located in your CWD"
                    )
                else:
                    ret = default_val
        else:  # treat `var` literally
            ret = var
        return ret

    dotenv.load_dotenv(override=True)  # load environment variables from .env file
    my_cfg = inference_option_list[inference_option]
    my_url = _get_var(my_cfg["url"], my_cfg["url_default"])
    my_token = _get_var(my_cfg["token"])
    zoo = dg.connect_model_zoo(my_url, my_token)  # connect to the model zoo
    print(f"Inference option = '{my_cfg['desc']}'")
    return zoo


@contextmanager
def open_video_stream(camera_id=None):
    """Open OpenCV video stream from camera with given identifier.

    camera_id - 0-based index for local cameras
       or IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>"

    Returns context manager yielding video stream object and closing it on exit
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

    try:
        yield stream
    finally:
        stream.release()


def video_source(stream, report_error=True):
    """Generator function, which returns video frames captured from given video stream.
    Useful to pass to model batch_predict().

    stream - video stream context manager object returned by open_video_stream()
    report_error - when True, error is raised on stream end

    Yields video frame captured from given video stream
    """
    while True:
        ret, frame = stream.read()
        if not ret:
            if report_error:
                raise Exception(
                    "Fail to capture camera frame. May be camera was opened by another notebook?"
                )
            else:
                break
        yield frame


@contextmanager
def open_audio_stream(sampling_rate_hz, buffer_size):
    """Open PyAudio audio stream

    sampling_rate_hz - desired sample rate in Hz
    buffer_size - read buffer size
    Returns context manager yielding audio stream object and closing it on exit
    """

    import numpy as np, queue

    try:
        import pyaudio
    except Exception as e:
        raise Exception(f"Error loading pyaudio package: {e}")

    audio = pyaudio.PyAudio()
    result_queue = queue.Queue()

    def callback(
        in_data,  # recorded data if input=True; else None
        frame_count,  # number of frames
        time_info,  # dictionary
        status_flags,
    ):  # PaCallbackFlags
        result_queue.put(in_data)
        return (None, pyaudio.paContinue)

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=int(sampling_rate_hz),
        input=True,
        frames_per_buffer=int(buffer_size),
        stream_callback=callback,
    )
    stream.result_queue = result_queue

    try:

        yield stream
    finally:
        stream.stop_stream()  # stop audio streaming
        stream.close()  # close audio stream
        audio.terminate()  # terminate audio library


def audio_source(stream, check_abort, non_blocking=False):
    """Generator function, which returns audio frames captured from given audio stream.
    Useful to pass to model batch_predict().

    stream - audio stream context manager object returned by open_audio_stream()
    check_abort - check-for-abort function or lambda; stream will be terminated when it returns True
    non_blocking - True for non-blocking mode (immediately yields None if a block is not captured yet)
        False for blocking mode (waits for the end of the block capture and always yields captured block)

    Yields audio waveform captured from given audio stream
    """

    import numpy as np, queue

    while not check_abort():
        if non_blocking:
            try:
                block = stream.result_queue.get_nowait()
            except queue.Empty:
                block = None
        else:
            block = stream.result_queue.get()

        yield None if block is None else np.frombuffer(block, dtype=np.int16)


def audio_overlapped_source(stream, check_abort, non_blocking=False):
    """Generator function, which returns audio frames captured from given audio stream with half-length overlap.
    Useful to pass to model batch_predict().

    stream - audio stream context manager object returned by open_audio_stream()
    check_abort - check-for-abort function or lambda; stream will be terminated when it returns True
    non_blocking - True for non-blocking mode (immediately yields None if a block is not captured yet)
        False for blocking mode (waits for the end of the block capture and always yields captured block)

    Yields audio waveform captured from given audio stream with half-length overlap.
    """

    import numpy as np, queue

    chunk_length = stream._frames_per_buffer
    data = np.zeros(2 * chunk_length, dtype=np.int16)
    while not check_abort():

        if non_blocking:
            try:
                block = stream.result_queue.get_nowait()
            except queue.Empty:
                block = None
        else:
            block = stream.result_queue.get()

        if block is None:
            yield None
        else:
            data[:chunk_length] = data[chunk_length:]
            data[chunk_length:] = np.frombuffer(block, dtype=np.int16)
            yield data


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
        """Record timestamp and update average duration.

        Returns current average FPS"""
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
        """Crop and return opencv image to given bbox"""
        return img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

    def put_text(
        img,
        text,
        position,
        text_color,
        back_color=None,
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ):
        """Draw given text on given image at given point with given color

        img - numpy array with image
        text - text to draw
        position - text top left coordinate tuple (x,y)
        text_color - text color (BGR)
        back_color - background color (BGR) or None for transparent
        font = font to use
        """

        text_size = cv2.getTextSize(text, font, 1, 1)
        text_w = text_size[0][0]
        text_h = text_size[0][1] + text_size[1]
        margin = int(text_h / 4)
        bl_corner = (position[0], position[1] + text_h)
        if back_color is not None:
            tr_corner = (
                bl_corner[0] + text_w + 2 * margin,
                bl_corner[1] - text_h - 2 * margin,
            )
            cv2.rectangle(img, bl_corner, tr_corner, back_color, cv2.FILLED)
        cv2.putText(
            img,
            text,
            (bl_corner[0] + margin, bl_corner[1] - margin),
            font,
            1,
            text_color,
        )

    def _check_gui():
        """Check if graphical display is supported

        Returns False if not supported
        """
        import os, platform

        if platform.system() == "Linux":
            return os.environ.get("DISPLAY") is not None
        return True

    def _show_fps(img, fps):
        """Helper method to display FPS"""
        Display.put_text(img, f"{fps:5.1f} FPS", (1, 1), (0, 0, 0), (255, 255, 255))

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


class Timer:
    """Simple timer class"""

    def __init__(self):
        """Constructor. Records start time."""
        self._start_time = time.time_ns()

    def __call__(self):
        """Call method.

        Returns time elapsed (in seconds, since object construction)."""
        return (time.time_ns() - self._start_time) * 1e-9
