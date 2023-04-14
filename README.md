# DeGirum PySDK Examples

## Quick Start

1. Create an account on [DeGirum Cloud Portal](https://cs.degirum.com). Use *Request Access* button on a main 
page to request access.

1. You should receive registration e-mail within one day. Follow instructions in e-mail to register your account.

1. Log in to [DeGirum Cloud Portal](https://cs.degirum.com).

1. Create cloud API access token on **My Tokens** page accessible via *Management > My Tokens* menu.

1. Install DeGirum PySDK. Read instructions on **General Information** page accessible via
*Documentation > General Information* menu.

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
    model = zoo.load_model(model_name)

    # perform AI inference of an image specified by URL
    image_url = "https://degirum.github.io/images/samples/TwoCats.jpg"
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
    |`AISERVER_HOSTNAME_OR_IP`|The hostname or IP address of a computer in your LAN/VPN which hosts AI Server. For localhost server, specify "localhost". Refer to *Documentation > General Information* page on [DeGirum Cloud Portal](https://cs.degirum.com) for AI server installation details.|
    |`CLOUD_ZOO_URL`|The cloud zoo URL to get models from. Format: `<organization>/<zoo>`. To confirm zoo URL visit *Management > Models* page on [DeGirum Cloud Portal](https://cs.degirum.com).|
    |`CAMERA_ID`|Local camera index, or web camera URL in the format `rtsp://<user>:<password>@<ip or hostname>`, or path to a video file.|

    This will allow loading the required information from the `env.ini` file instead of hard-coding the values in the script. 

    The `env.ini` file is added to `.gitignore` and will not be checked in. This will ensure that your token information 
    is not leaked. 

## Examples Directory

* [Basic Examples](#basic-examples)
* [Combining Multiple Models](#combining-multiple-models)
* [Advanced Algorithms](#advanced-algorithms)
* [Operating with Datasets](#operating-with-datasets)
* [Sound Processing](#sound-processing)
* [Benchmarks](#benchmarks)

### Basic Examples

| Example | Description |
|---------|-------------|
|[mystreamsDemo](https://github.com/DeGirum/PySDKExamples/blob/main/mystreamsDemo.ipynb)|Extensive demo notebook of `mystreams` toolkit: lightweight multi-threaded pipelining framework|
|[ObjectDetectionImage](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionImage.ipynb)|One of the most simplest examples how to do AI inference of a graphical file using object detection model.|
|[ObjectDetectionVideoFile](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionVideoFile.ipynb)|How to do AI inference of a video stream from a video file, show annotated video, and save it to another video file. |
|[ObjectDetectionCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionCameraStream.ipynb)|How to do AI inference of a video stream from a video camera and show annotated video in real-time.|
|[ObjectDetectionVideoFile2Images](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionVideoFile2Images.ipynb)|How to do AI inference of a video stream from a video file and save annotated frame images into a directory.|


### Combining Multiple Models

| Example | Description |
|---------|-------------|
|[PersonPoseDetection PipelinedImage](https://github.com/DeGirum/PySDKExamples/blob/main/PersonPoseDetectionPipelinedImage.ipynb)|How to do AI inference of a graphical file using two AI models: person detection and pose detection. The person detection model is run on the image and the results are then processed by the pose detection model, one person bounding box at a time. Combined result is then displayed.|
|[PersonPoseDetection PipelinedCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/PersonPoseDetectionPipelinedCameraStream.ipynb)|A video stream from a video camera is processed by the person detection model. The person detection results are then processed by the pose detection model, one person bounding box at a time. Combined results are then displayed as an annotated video in real-time.|
|[FaceMaskDetection PipelinedImage](https://github.com/DeGirum/PySDKExamples/blob/main/FaceMaskDetectionPipelinedImage.ipynb)|How to do AI inference of a graphical file using two AI models: face detection and mask detection. The face detection model is run on the image and the results are then processed by the mask detection model, one face bounding box at a time. Combined result is then displayed.|
|[FaceMaskDetection PipelinedCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/FaceMaskDetectionPipelinedCameraStream.ipynb)|A video stream from a video camera is processed by the face detection model. The face detection results are then processed by the mask detection model, one face bounding box at a time. Combined results are then displayed as an annotated video in real-time.|
|[FaceHandDetection ParallelCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/FaceHandDetectionParallelCameraStream.ipynb)|How to run two models side-by-side and combine results of both models. A video stream from a video camera is processed simultaneously by the hand and face detection models. Combined result is then displayed.|
|[MultiCamera MultiModelDetection](https://github.com/DeGirum/PySDKExamples/blob/main/MultiCameraMultiModelDetection.ipynb)|How to perform AI inferences of multiple models processing multiple video streams. Each video stream is fed to every model. Each model processes frames from every video stream in multiplexing manner.|


### Advanced Algorithms

| Example | Description |
|---------|-------------|
|[TiledObjectDetectionVideoFile](https://github.com/DeGirum/PySDKExamples/blob/main/TiledObjectDetectionVideoFile.ipynb)|How to do tiled object detection of a video stream from a video file. Each video frame is divided by tiles with some overlap, each tile of the AI model input size (to avoid resizing). Object detection is performed for each tile, then results from different tiles are combined. When motion detection mode is enabled, object detection is performed only for tiles where motion is detected.|
|[MultiObjectTrackingVideoFile](https://github.com/DeGirum/PySDKExamples/blob/main/MultiObjectTrackingVideoFile.ipynb)|How to perform object detection with multi-object tracking (MOT) from a video file to count vehicle traffic.|


### Operating with Datasets

| Example | Description |
|---------|-------------|
|[ObjectDetectionDataset](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionDataset.ipynb)|How to do AI inference on an image dataset and calculate performance metrics. An image dataset is retrieved from the cloud using `fiftyone` API.|
|[ObjectDetection DatasetMultithreaded](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionDatasetMultithreaded.ipynb)|How to do **multi-threaded** AI inference on an image dataset. An image dataset is retrieved from the cloud using `fiftyone` API.|

### Sound Processing

| Example | Description |
|---------|-------------|
|[SoundClassificationAudioStream](https://github.com/DeGirum/PySDKExamples/blob/main/SoundClassificationAudioStream.ipynb)|How to do sound classification AI inference of an audio stream from a local microphone in real time. The result label with highest probability is displayed for each inference while keeping history few steps back.|
|[SoundClassificationAnd ObjectDetectionAsynchronous](https://github.com/DeGirum/PySDKExamples/blob/main/SoundClassificationAndObjectDetectionAsynchronous.ipynb)|How to perform parallel inferences on two **asynchronous** data streams with different frame rates. To achieve maximum performance this example uses **non-blocking** batch prediction mode.|

### Benchmarks

| Example | Description |
|---------|-------------|
|[SingleModelPerformaceTest](https://github.com/DeGirum/PySDKExamples/blob/main/SingleModelPerformaceTest.ipynb)|Performance measurements for all Orca-based image detection AI models from DeGirum public model zoo.|
|[MultiModelPerformaceTest](https://github.com/DeGirum/PySDKExamples/blob/main/MultiModelPerformaceTest.ipynb)|Performance measurements for simultaneous inference of multiple AI models.|
|[ObjectDetection MultiplexingMultipleStreams](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionMultiplexingMultipleStreams.ipynb)|How to perform object detection from multiple video files, multiplexing frames. This example demonstrates lowest possible and stable AI inference latency while maintaining decent throughput. This is achieved by using synchronous prediction mode and video decoding offloaded into separate thread.|
