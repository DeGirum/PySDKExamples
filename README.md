# DeGirum PySDK Examples

**[ORCA1 Performance Benchmarks](ORCABenchmarks.md)**

## Quick Start

1. Sign up for an account on [DeGirum Cloud Portal](https://cs.degirum.com). 

1. Log in to [DeGirum Cloud Portal](https://cs.degirum.com).

1. Create cloud API access token on **My Tokens** page accessible via *Management > My Tokens* menu.

1. Install DeGirum PySDK. Read the instructions [here](https://docs.degirum.com/content/pysdk/installation/).

1. The following script will download *MobileNetv2+SSD* CPU model from 
DeGirum public mode zoo and perform ML inference of a test image with two cats. 
The inference result will be displayed in both text and graphical form.

    ```python
    import degirum as dg         # import DeGirum PySDK package
    # connect to DeGirum cloud platform and use DeGirum public model zoo
    zoo = dg.connect(dg.CLOUD, "https://cs.degirum.com", "<my cloud API access token>")
    print(zoo.list_models())     # print all available models in the model zoo

    # load mobilenet_ssd model for CPU; model_name should be one returned by zoo.list_models()
    model_name = "mobilenet_v2_ssd_coco--300x300_quant_n2x_cpu_1"     
    model = zoo.load_model(model_name, image_backend='pil')

    # perform AI inference of an image specified by URL
    image_url = "https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/TwoCats.jpg"
    result = model(image_url)

    print(result)                # print numeric results
    result.image_overlay.show()  # show graphical results
    ```

## Running PySDK Examples

This repository provides PySDK example scripts that can perform ML inferences on the following hosting options:

1. Using [DeGirum Cloud Platform](https://cs.degirum.com),
1. On DeGirum AI Server deployed on a localhost or on some computer in your LAN or VPN,
1. On DeGirum ORCA accelerator directly installed on your local computer.

To try different options, you just need to uncomment **one** of the lines in the code cell just below the
*"Specify where do you want to run your inferences"* header.

To run examples, please perform the following steps:

1. Make sure you have installed Python version 3.9, 3.10, or 3.11. For convenience of future maintenance we recommend 
you to work in the virtual environment, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 
Make sure you activated your Python virtual environment.

1. Clone DeGirum PySDKExamples repo by executing the following command in the terminal / command prompt:

    ```
    git clone https://github.com/DeGirum/PySDKExamples.git
    ```

1. In the terminal / command prompt, change the current directory to the repo directory, and install necessary Python
dependencies by executing the following command:

    ``` Python
    pip install -r requirements.txt
    ```

1. Inside the repo directory, open `env.ini` file and fill the required authentication details by assigning the
following variables:

    |Variable Name|Description|
    |-------------|-----------|
    |`DEGIRUM_CLOUD_TOKEN`|DeGirum cloud platform API access token. To obtain a token, visit *Management > My Tokens* page on [DeGirum Cloud Portal](https://cs.degirum.com).|

    This will allow loading the token from the `env.ini` file instead of hard-coding the value in the script. The `env.ini` file is added to `.gitignore` and will not be checked in. This will ensure that your token information 
    is not leaked. 

## Examples Directory

* [Basic Examples](#basic-examples)
* [Combining Multiple Models](#combining-multiple-models)
* [Advanced Algorithms](#advanced-algorithms)
* [Benchmarks](#benchmarks)
* [Examples of `degirum_tools.streams` Toolkit Usage](#examples-of-degirum_toolsstreams-toolkit-usage)

### Basic Examples

| Example | Description |
|---------|-------------|
|[object detection image](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/basic/object_detection_image.ipynb)|One of the most simplest examples how to do AI inference on a graphical file using object detection model.|
|[object detection video file](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/basic/object_detection_video_file.ipynb)|How to do AI inference on a video file, show annotated video, and save it to another video file. |
|[object detection video stream](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/basic/object_detection_video_stream.ipynb)|How to do AI inference on a video stream, and show annotated video in real-time.|
|[sound classification audio stream](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/basic/sound_classification_audio_stream.ipynb)|How to do sound classification AI inference on an audio stream from a local microphone in real time. The result label with highest probability is displayed for each inference while keeping history few steps back.|


### Combining Multiple Models

| Example | Description |
|---------|-------------|
|[hand face person detection parallel camera stream](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/multimodel/hand_face_person_detection_parallel_video_stream.ipynb)|How to run three models side-by-side and combine results of the three models. A video stream is processed simultaneously by the hand, face, and person detection models. Combined result is then displayed.|
|[license plate recognition pipelined image](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/multimodel/license_plate_recognition_pipelined_image.ipynb)|How to do AI inference of a graphical file using two AI models: license plate detection and license plate text recognition. The license plate detection model is run on the image and the results are then processed by the license plate text recognition model, one bounding box at a time. Combined result is then displayed.|
|[license plate recognition pipelined camera stream](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/multimodel/license_plate_recognition_pipelined_camera_stream.ipynb)|A video stream from a video camera is processed by the license plate detection model. The face detection results are then processed by the license plate text recognition model, one bounding box at a time. Combined results are then displayed as an annotated video in real-time.|
|[sound classification and object detection asynchronous](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/multimodel/sound_classification_and_object_detection_asynchronous.ipynb)|How to perform parallel inferences on two **asynchronous** data streams with different frame rates. To achieve maximum performance this example uses **non-blocking** batch prediction mode.|

### Advanced Algorithms

| Example | Description |
|---------|-------------|
|[multi object tracking video file](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/specialized/multi_object_tracking_video_file.ipynb)|How to perform object detection with multi-object tracking (MOT) from a video file to count vehicle traffic.|
|[sliced object detection](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/specialized/sliced_object_detection.ipynb)|How to do sliced object detection of a video stream from a video file. Each video frame is divided by slices/tiles with some overlap, each tile of the AI model input size (to avoid resizing). Object detection is performed for each tile, then results from different tiles are combined. When motion detection mode is enabled, object detection is performed only for tiles where motion is detected.|
|[object in zone counting video stream](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/specialized/object_in_zone_counting_video_stream.ipynb)|Object detection and object counting in polygon zone: streaming video processing|
|[object in zone counting video file](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/specialized/object_in_zone_counting_video_file.ipynb)|Object detection and object counting in polygon zone: video file annotation|

### Benchmarks

| Example | Description |
|---------|-------------|
|[single model performance test](https://githubcolab.com/DeGirum/PySDKExamples/blob/main/examples/benchmarks/single_model_performance_test.ipynb)|Performance measurements for all Orca-based image detection AI models from DeGirum public model zoo.|
|[multi model performance test](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/benchmarks/multi_model_performance_test.ipynb)|Performance measurements for simultaneous inference of multiple AI models.|
|[object detection multiplexing multiple streams](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/benchmarks/object_detection_multiplexing_multiple_streams.ipynb)|How to perform object detection from multiple video files, multiplexing frames. This example demonstrates lowest possible and stable AI inference latency while maintaining decent throughput. This is achieved by using synchronous prediction mode and video decoding offloaded into separate thread.|

### Examples of `degirum_tools.streams` Toolkit Usage

| Example | Description |
|---------|-------------|
|[dgstreams demo](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/dgstreams/dgstreams_demo.ipynb)|Extensive demo notebook of `degirum_tools.streams` toolkit: lightweight multi-threaded pipelining framework|
|[multi camera multi model detection](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/dgstreams/multi_camera_multi_model_detection.ipynb)|How to perform AI inferences of multiple models processing multiple video streams. Each video stream is fed to every model. Each model processes frames from every video stream in multiplexing manner.|
|[person pose detection pipelined camera stream](https://colab.research.google.com/github/DeGirum/PySDKExamples/blob/main/examples/dgstreams/person_pose_detection_pipelined_video_stream.ipynb)|A video stream from a video camera is processed by the person detection model. The person detection results are then processed by the pose detection model, one person bounding box at a time. Combined results are then displayed as an annotated video in real-time.|
