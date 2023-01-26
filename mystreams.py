#
# mystreams.py: streaming toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Please refer to `mystreamDemo.ipynb` notebook for examples of toolkit usage.
#

import threading
import queue
import copy
from abc import ABC, abstractmethod
from typing import Optional, Any, List

import cv2
import numpy

import degirum as dg
import mytools


class StreamData:
    """Single data element of the streaming pipelines"""

    def __init__(self, data: Any, meta: Any):
        """Constructor.

        - data: data payload
        - meta: metainfo"""
        self.data = data
        self.meta = meta


class Stream(queue.Queue):
    """Queue-based iterable class with optional item drop"""

    def __init__(self, maxsize=0, allow_drop: bool = False):
        """Constructor

        - maxsize: maximum stream depth; 0 for unlimited depth
        - allow_drop: allow dropping elements on put() when stream is full
        """
        super().__init__(maxsize)
        self._allow_drop = allow_drop

    _poison = None

    def put(self, item: Optional[StreamData]):
        """Put an item into the stream

        - item: item to put
        If there is no space left, and allow_drop flag is set, then oldest item will
        be popped to free space
        """
        if self._allow_drop:
            while True:
                try:
                    super().put(item, False)
                    break
                except queue.Full:
                    try:
                        self.get_nowait()
                    finally:
                        pass
        else:
            super().put(item)

    def __iter__(self):
        """Iterator method"""
        return iter(self.get, self._poison)

    def close(self):
        """Close stream: put poison pill"""
        self.put(self._poison)


class Gizmo(ABC):
    """Base class for all gizmos: streaming pipeline processing blocks.
    Each gizmo owns zero of more input streams, which are used to deliver
    the data to that gizmo for processing. For data-generating gizmos
    there is no input stream.

    A gizmo can be connected to other gizmo to receive a data from that
    gizmo into one of its own input streams. Multiple gizmos can be connected to
    a single gizmo, so one gizmo can broadcast data to multiple destinations.

    A data element is a tuple containing raw data object as a first element, and meta info
    object as a second element.

    Abstract run() method should be implemented in derived classes to run gizmo
    processing loop. It is not called directly, but is launched by Composition class
    in a separate thread.

    run() implementation should periodically check for _abort flag (set by abort())
    and run until there will be no more data in its input(s).

    """

    def __init__(self, input_stream_sizes: List[tuple] = []):
        """Constructor

        - input_stream_size: a list of tuples containing constructor parameters of input streams;
            pass empty list to have no inputs; zero means unlimited depth
        """

        self._inputs = []
        for s in input_stream_sizes:
            self._inputs.append(Stream(*s))

        self._output_refs = []
        self._abort = False
        self.composition: Optional[Composition] = None

    def get_input(self, inp: int) -> Stream:
        """Get inp-th input stream"""
        if inp >= len(self._inputs):
            raise Exception(f"Input {inp} is not assigned")
        return self._inputs[inp]

    def connect_to(self, other_gizmo, inp: int = 0):
        """Connect given input to other gizmo.

        - other_gizmo: gizmo to connect to
        - inp: input index to use for connection
        """
        other_gizmo._output_refs.append(self.get_input(inp))

    def __lshift__(self, other_gizmo):
        """Operator synonym for connect_to(): connects self to other_gizmo
        Returns other_gizmo"""
        self.connect_to(other_gizmo)
        return other_gizmo

    def __rshift__(self, other_gizmo):
        """Operator antonym for connect_to(): connects other_gizmo to self

        Returns self"""
        other_gizmo.connect_to(self)
        return other_gizmo

    def send_result(self, data: Optional[StreamData], clone_data: bool = False):
        """Send result to all connected outputs.

        - data: a tuple containing raw data object as a first element, and meta info object as a second element.
        When None is passed, all outputs will be closed.
        - clone_data: clone data when sending to different outputs
        """
        for out in self._output_refs:
            if data is None:
                out.close()
            else:
                out.put(copy.deepcopy(data) if clone_data else data)

    @abstractmethod
    def run(self):
        """Run gizmo"""

    def abort(self, abort: bool = True):
        """Set abort flag"""
        self._abort = abort


class Composition:
    """Class, which holds and animates multiple connected gizmos.
    First you add all necessary gizmos to your composition using add() or __call()__ method.
    Then you connect all added gizmos in proper order using connect_to() method or `>>` operator.
    Then you start your composition by calling start() method.
    You may stop your composition by calling stop() method."""

    def __init__(self):
        """Constructor."""
        self._gizmos = []
        self._treads = []

    def add(self, gizmo: Gizmo) -> Gizmo:
        """Add a gizmo to composition

        - gizmo: gizmo to add

        Returns same gizmo
        """
        gizmo.composition = self
        self._gizmos.append(gizmo)
        return gizmo

    def __call__(self, gizmo: Gizmo) -> Gizmo:
        """Operator synonym for add()"""
        return self.add(gizmo)

    def start(self):
        """Start gizmo animation: launch run() method of every registered gizmo in a separate thread"""

        if len(self._treads) > 0:
            raise Exception("Composition already started")

        for gizmo in self._gizmos:
            gizmo.abort(False)
            self._treads.append(threading.Thread(target=gizmo.run))

        for t in self._treads:
            t.start()

        print("Composition started")

    def stop(self):
        """Signal abort to all registered gizmos and wait until all threads stopped"""

        if len(self._treads) == 0:
            raise Exception("Composition not started")

        for gizmo in self._gizmos:
            gizmo.abort()
            for i in gizmo._inputs:
                i.close()

        def do_join():
            for t in self._treads:
                t.join()

        threading.Thread(target=do_join).start()

        self._treads = []
        print("Composition stopped")


class VideoSourceGizmo(Gizmo):
    """OpenCV-based video source gizmo"""

    def __init__(self, video_source=0):
        """Constructor.

        - video_source: cv2.VideoCapture-compatible video source designator
        """
        super().__init__()
        self._video_source = video_source

    def run(self):
        """Run gizmo"""
        with mytools.open_video_stream(self._video_source) as src:
            while not self._abort:
                ret, data = src.read()
                if not ret:
                    self._abort = True
                    self.send_result(None)
                else:
                    self.send_result(StreamData(data, {}))


class VideoDisplayGizmo(Gizmo):
    """OpenCV-based video display gizmo"""

    def __init__(
        self,
        window_title: str = "",
        *,
        show_ai_overlay=False,
        show_fps: bool = False,
        stream_depth: int = 10,
        allow_drop: bool = True,
    ):
        """Constructor.

        - window_title: window title string
        - show_fps: True to show FPS
        - show_ai_overlay: True to show AI inference overlay image when possible
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._window_title = window_title
        self._show_fps = show_fps
        self._show_ai_overlay = show_ai_overlay

    def run(self):
        """Run gizmo"""
        first_run = True
        with mytools.Display(self._window_title, self._show_fps) as display:
            try:
                for data in self.get_input(0):
                    if self._abort:
                        break

                    if self._show_ai_overlay and isinstance(
                        data.meta, dg.postprocessor.InferenceResults
                    ):
                        # show AI inference overlay if possible
                        display.show(data.meta.image_overlay)
                    else:
                        display.show(data.data)

                    if first_run:
                        cv2.setWindowProperty(
                            self._window_title, cv2.WND_PROP_TOPMOST, 1
                        )
                        first_run = False

            except KeyboardInterrupt:
                if self.composition is not None:
                    self.composition.stop()


class VideoSaverGizmo(Gizmo):
    """OpenCV-based gizmo to save video to a file"""

    def __init__(
        self,
        filename: str = "",
        *,
        show_ai_overlay=False,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - filename: output video file name
        - show_ai_overlay: True to show AI inference overlay image when possible
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._filename = filename
        self._show_ai_overlay = show_ai_overlay

    def run(self):
        """Run gizmo"""

        get_img = (
            lambda data: data.meta.image_overlay
            if self._show_ai_overlay
            and isinstance(data.meta, dg.postprocessor.InferenceResults)
            else data.data
        )

        img = get_img(self.get_input(0).get())
        with mytools.open_video_writer(
            self._filename, img.shape[1], img.shape[0]
        ) as writer:
            writer.write(img)
            for data in self.get_input(0):
                if self._abort:
                    break
                writer.write(get_img(data))


class ResizingGizmo(Gizmo):
    """OpenCV-based image resizing/padding gizmo"""

    def __init__(
        self,
        w: int,
        h: int,
        pad_method: str = "letterbox",
        resize_method: int = cv2.INTER_LINEAR,
    ):
        """Constructor.

        - w, h: resulting image width/height
        - pad_method: padding method - one of 'stretch', 'letterbox'
        - resize_method: resampling method - one of cv2.INTER_xxx constants
        """
        super().__init__([(0, False)])
        self._h = w
        self._w = w
        self._pad_method = pad_method
        self._resize_method = resize_method

    def _resize(self, image):
        dx = dy = 0  # offset of left top corner of original image in resized image

        image_ret = image
        if image.shape[1] != self._w or image.shape[0] != self._h:
            if self._pad_method == "stretch":
                image_ret = cv2.resize(
                    image, (self._w, self._h), interpolation=self._resize_method
                )
            elif self._pad_method == "letterbox":
                iw = image.shape[1]
                ih = image.shape[0]
                scale = min(self._w / iw, self._h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)

                # resize preserving aspect ratio
                scaled_image = cv2.resize(
                    image, (nw, nh), interpolation=self._resize_method
                )

                # create new canvas image and paste into it
                image_ret = numpy.zeros((self._h, self._w, 3), image.dtype)

                dx = (self._w - nw) // 2
                dy = (self._h - nh) // 2
                image_ret[dy : dy + nh, dx : dx + nw, :] = scaled_image

        return image_ret

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break
            resized = self._resize(data.data)
            self.send_result(StreamData(resized, data.meta))


class AiGizmoBase(Gizmo):
    """Base class for AI inference gizmos"""

    def __init__(self, model, *, stream_depth: int = 10, allow_drop: bool = False):
        """Constructor.

        - model: PySDK model object
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])

        model.image_backend = "opencv"  # select OpenCV backend
        model.input_numpy_colorspace = "BGR"  # adjust colorspace to match OpenCV
        self.model = model

    def run(self):
        """Run gizmo"""

        def source():
            for data in self.get_input(0):
                yield (data.data, data)

        for result in self.model.predict_batch(source()):
            self.on_result(result)
            if self._abort:
                break

    @abstractmethod
    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference"""


class AiSimpleGizmo(AiGizmoBase):
    """AI inference gizmo with no result processing: it passes through input frames
    attaching inference results as meta info"""

    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference"""
        self.send_result(StreamData(result.image, result))


class AiObjectDetectionCroppingGizmo(AiGizmoBase):
    """AI object detection inference gizmo which sends crops of each detected object.
    Output meta-info is a dictionary with the following keys:

    - "original_result": reference to original AI object detection result (contained only in the first crop)
    - "cropped_result": reference to sub-result for particular crop
    - "cropped_index": the number of that sub-result
    The last two key are present only if at least one object is detected in a frame.
    """

    def __init__(self, labels: List[str], *args, **kwargs):
        """Constructor.

        - labels: list of class labels to process
        """
        super().__init__(*args, **kwargs)
        self._labels = labels

    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference"""

        if len(result.results) == 0:  # no objects detected
            self.send_result(StreamData(result.image, {"original_result": result}))

        is_first = True
        for i, r in enumerate(result.results):
            if r["label"] not in self._labels:
                continue
            crop = mytools.Display.crop(result.image, r["bbox"])
            # send all crops afterwards
            meta = {}
            if is_first:
                # send whole result with no data first
                meta["original_result"] = result
                is_first = False

            meta["cropped_result"] = r
            meta["cropped_index"] = i
            self.send_result(StreamData(crop, meta))
