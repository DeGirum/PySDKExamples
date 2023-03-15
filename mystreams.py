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
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Union
from contextlib import ExitStack

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
        self.allow_drop = allow_drop
        self.dropped_cnt = 0  # number of dropped items

    _poison = None

    def put(self, item: Optional[StreamData]):
        """Put an item into the stream

        - item: item to put
        If there is no space left, and allow_drop flag is set, then oldest item will
        be popped to free space
        """
        if self.allow_drop:
            while True:
                try:
                    super().put(item, False)
                    break
                except queue.Full:
                    self.dropped_cnt += 1
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

        self._inputs: List[Stream] = []
        for s in input_stream_sizes:
            self._inputs.append(Stream(*s))

        self._output_refs = []
        self._abort = False
        self.composition: Optional[Composition] = None
        self.name = self.__class__.__name__
        self.result_cnt = 0  # gizmo result counter
        self.start_time_s = time.time()  # gizmo start time
        self.elapsed_s = 0
        self.fps = 0  # achieved FPS rate

    def get_input(self, inp: int) -> Stream:
        """Get inp-th input stream"""
        if inp >= len(self._inputs):
            raise Exception(f"Input {inp} is not assigned")
        return self._inputs[inp]

    def get_inputs(self) -> List[Stream]:
        """Get list of input streams"""
        return self._inputs

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
        self.result_cnt += 1
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
        self._gizmos: List[Gizmo] = []
        self._threads: List[threading.Thread] = []

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

    def start(self, detect_bottlenecks: bool = False):
        """Start gizmo animation: launch run() method of every registered gizmo in a separate thread.

        - detect_bottlenecks: true to switch all streams into dropping mode to detect bottlenecks.
        Use get_bottlenecks() method to return list of gizmos-bottlenecks
        """

        if len(self._threads) > 0:
            raise Exception("Composition already started")

        def gizmo_run(gizmo):
            gizmo.result_cnt = 0
            if detect_bottlenecks:
                for i in gizmo.get_inputs():
                    i.allow_drop = True
            gizmo.start_time_s = time.time()
            gizmo.run()
            gizmo.elapsed_s = time.time() - gizmo.start_time_s
            gizmo.fps = gizmo.result_cnt / gizmo.elapsed_s if gizmo.elapsed_s > 0 else 0
            gizmo.send_result(Stream._poison)

        for gizmo in self._gizmos:
            gizmo.abort(False)
            t = threading.Thread(target=gizmo_run, args=(gizmo,))
            t.name = t.name + "-" + type(gizmo).__name__
            self._threads.append(t)

        for t in self._threads:
            t.start()

        print("Composition started")

        # test mode has limited inputs
        if mytools.get_test_mode():
            self.wait()

    def get_bottlenecks(self) -> List[str]:
        """Return a list of gizmo names, which experienced bottlenecks during last run.
        Composition should be started with detect_bottlenecks=True to use this feature.
        """
        ret = []
        for gizmo in self._gizmos:
            for i in gizmo.get_inputs():
                if i.dropped_cnt > 0:
                    ret.append({gizmo.name: i.dropped_cnt})
                    break
        return ret

    def stop(self):
        """Signal abort to all registered gizmos and wait until all threads stopped"""

        if len(self._threads) == 0:
            raise Exception("Composition not started")

        def do_join():

            # first signal abort to all gizmos
            for gizmo in self._gizmos:
                gizmo.abort()

            # then empty all streams
            for gizmo in self._gizmos:
                for i in gizmo._inputs:
                    while not i.empty():
                        try:
                            i.get_nowait()
                        except queue.Empty:
                            break

            # finally wait for completion of all threads
            for t in self._threads:
                t.join()

        # do it in a separate thread, because stop() may be called by some gizmo
        threading.Thread(target=do_join).start()

        self._threads = []
        print("Composition stopped")

    def wait(self):
        """Wait until all threads stopped"""

        if len(self._threads) == 0:
            raise Exception("Composition not started")

        for t in self._threads:
            t.join()


class VideoSourceGizmo(Gizmo):
    """OpenCV-based video source gizmo"""

    def __init__(self, video_source=None):
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
                else:
                    self.send_result(StreamData(data, {}))


class VideoDisplayGizmo(Gizmo):
    """OpenCV-based video display gizmo"""

    def __init__(
        self,
        window_titles: Union[str, List[str]] = "Display",
        *,
        show_ai_overlay: bool = False,
        show_fps: bool = False,
        stream_depth: int = 10,
        allow_drop: bool = False,
        multiplex: bool = False,
    ):
        """Constructor.

        - window_titles: window title string or array of title strings for multiple displays
        - show_fps: True to show FPS
        - show_ai_overlay: True to show AI inference overlay image when possible
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        - multiplex: then True, single input data is multiplexed to multiple displays;
          when False, each input is displayed on individual display
        """

        if isinstance(window_titles, str):
            window_titles = [
                window_titles,
            ]

        inp_cnt = 1 if multiplex else len(window_titles)
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)
        self._window_titles = window_titles
        self._show_fps = show_fps
        self._show_ai_overlay = show_ai_overlay
        self._multiplex = multiplex

    def run(self):
        """Run gizmo"""

        with ExitStack() as stack:

            ndisplays = len(self._window_titles)
            ninputs = len(self.get_inputs())

            displays = [
                stack.enter_context(mytools.Display(w, self._show_fps))
                for w in self._window_titles
            ]
            first_run = [True] * ndisplays

            di = 0  # di is display index
            try:
                while True:
                    if self._abort:
                        break

                    for ii, input in enumerate(self.get_inputs()):  # ii is input index
                        try:
                            if ninputs > 1:
                                # non-multiplexing multi-input case
                                data = input.get_nowait()
                            else:
                                # single input or multiplexing case
                                data = input.get()
                                self.result_cnt += 1
                        except queue.Empty:
                            continue

                        # select display to show this frame
                        di = (di + 1) % ndisplays if self._multiplex else ii

                        if data == Stream._poison:
                            self._abort = True
                            break

                        if self._show_ai_overlay and isinstance(
                            data.meta, dg.postprocessor.InferenceResults
                        ):
                            # show AI inference overlay if possible
                            displays[di].show(data.meta.image_overlay)
                        else:
                            displays[di].show(data.data)

                        if first_run[di] and not displays[di]._no_gui:
                            cv2.setWindowProperty(
                                self._window_titles[di], cv2.WND_PROP_TOPMOST, 1
                            )
                            first_run[di] = False

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
            self.result_cnt += 1
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
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - w, h: resulting image width/height
        - pad_method: padding method - one of 'stretch', 'letterbox'
        - resize_method: resampling method - one of cv2.INTER_xxx constants
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
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

    def __init__(
        self,
        model,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
        inp_cnt: int = 1,
    ):
        """Constructor.

        - model: PySDK model object
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)

        model.image_backend = "opencv"  # select OpenCV backend
        model.input_numpy_colorspace = "BGR"  # adjust colorspace to match OpenCV
        self.model = model

    def run(self):
        """Run gizmo"""

        def source():

            while True:
                # get data from all inputs
                for inp in self.get_inputs():
                    d = inp.get()
                    if d == Stream._poison:
                        self._abort = True
                        break
                    yield (d.data, d)

                if self._abort:
                    break

        for result in self.model.predict_batch(source()):
            self.on_result(result)
            # finish processing all frames for tests
            if self._abort and not mytools.get_test_mode():
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


class AiResultCombiningGizmo(Gizmo):
    """Gizmo to combine results from multiple AI gizmos with similar-typed results"""

    def __init__(
        self,
        inp_cnt: int,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - inp_cnt: number of inputs to combine
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        self._inp_cnt = inp_cnt
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)

    def run(self):
        """Run gizmo"""

        while True:
            # get data from all inputs
            all_data = []
            for inp in self.get_inputs():
                d = inp.get()
                if d == Stream._poison:
                    self._abort = True
                    break
                all_data.append(d)

            if self._abort:
                break

            d0 = all_data[0]
            for d in all_data[1:]:
                # check that results are of the same type and from the same data
                if type(d.meta) == type(d0.meta):
                    d0.meta._inference_results += d.meta._inference_results

            self.send_result(StreamData(d0.data, d0.meta))


class FanoutGizmo(Gizmo):
    """Gizmo to fan-out single input into multiple outputs.
    NOTE: by default it drops frames when experiencing back-pressure."""

    def __init__(
        self,
        stream_depth: int = 2,
        allow_drop: bool = True,
    ):
        """Constructor.

        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])

    def run(self):
        """Run gizmo"""
        for data in self.get_input(0):
            if self._abort:
                break
            self.send_result(data)
