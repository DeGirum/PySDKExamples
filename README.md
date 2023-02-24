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
    zoo = dg.connect_model_zoo("dgcps://cs.degirum.com", token="<my cloud API access token>")
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

To run the examples, clone this repo:

```
git clone https://github.com/DeGirum/PySDKExamples.git
```

Inside the repo, open `env.ini` file and fill the required authentication details by assigning the following variables:

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

| Example | Description |
|---------|-------------|
|[mystreamsDemo](https://github.com/DeGirum/PySDKExamples/blob/main/mystreamsDemo.ipynb)|Extensive demo notebook of `mystreams` toolkit: lightweight multi-threaded pipelining framework|
|[ObjectDetectionImage](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionImage.ipynb)|One of the most simplest examples how to do AI inference of a graphical file using object detection model.|
|[ObjectDetectionVideoFile](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionVideoFile.ipynb)|How to do AI inference of a video stream from a video file, show annotated video, and save it to another video file. |
|[ObjectDetectionCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionCameraStream.ipynb)|How to do AI inference of a video stream from a video camera and show annotated video in real-time.|
|[ObjectDetectionDataset](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionDataset.ipynb)|How to do AI inference on an image dataset and calculate performance metrics. An image dataset is retrieved from the cloud using `fiftyone` API.|
|[ObjectDetectionDatasetMultithreaded](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionDatasetMultithreaded.ipynb)|How to do **multi-threaded** AI inference on an image dataset. An image dataset is retrieved from the cloud using `fiftyone` API.|
|[ObjectDetectionVideoFile2Images](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionVideoFile2Images.ipynb)|How to do AI inference of a video stream from a video file and save annotated frame images into a directory.|
|[PersonPoseDetectionPipelinedImage](https://github.com/DeGirum/PySDKExamples/blob/main/PersonPoseDetectionPipelinedImage.ipynb)|How to do AI inference of a graphical file using two AI models: person detection and pose detection. The person detection model is run on the image and the results are then processed by the pose detection model, one person bounding box at a time. Combined result is then displayed.|
|[PersonPoseDetectionPipelinedCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/PersonPoseDetectionPipelinedCameraStream.ipynb)|A video stream from a video camera is processed by the person detection model. The person detection results are then processed by the pose detection model, one person bounding box at a time. Combined results are then displayed as an annotated video in real-time.|
|[FaceMaskDetectionPipelinedImage](https://github.com/DeGirum/PySDKExamples/blob/main/FaceMaskDetectionPipelinedImage.ipynb)|How to do AI inference of a graphical file using two AI models: face detection and mask detection. The face detection model is run on the image and the results are then processed by the mask detection model, one face bounding box at a time. Combined result is then displayed.|
|[FaceMaskDetectionPipelinedCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/FaceMaskDetectionPipelinedCameraStream.ipynb)|A video stream from a video camera is processed by the face detection model. The face detection results are then processed by the mask detection model, one face bounding box at a time. Combined results are then displayed as an annotated video in real-time.|
|[FaceHandDetectionParallelCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/FaceHandDetectionParallelCameraStream.ipynb)|How to run two models side-by-side and combine results of both models. A video stream from a video camera is processed simultaneously by the hand and face detection models. Combined result is then displayed.|
|[SoundClassificationAudioStream](https://github.com/DeGirum/PySDKExamples/blob/main/SoundClassificationAudioStream.ipynb)|How to do sound classification AI inference of an audio stream from a local microphone in real time. The result label with highest probability is displayed for each inference while keeping history few steps back.|
|[SoundClassificationAndObjectDetectionAsynchronous](https://github.com/DeGirum/PySDKExamples/blob/main/SoundClassificationAndObjectDetectionAsynchronous.ipynb)|How to perform parallel inferences on two **asynchronous** data streams with different frame rates. To achieve maximum performance this example uses **non-blocking** batch prediction mode.|
|[TiledObjectDetectionVideoFile](https://github.com/DeGirum/PySDKExamples/blob/main/TiledObjectDetectionVideoFile.ipynb)|How to do tiled object detection of a video stream from a video file. Each video frame is divided by tiles with some overlap, each tile of the AI model input size (to avoid resizing). Object detection is performed for each tile, then results from different tiles are combined. When motion detection mode is enabled, object detection is performed only for tiles where motion is detected.|
|[MultiObjectTrackingVideoFile](https://github.com/DeGirum/PySDKExamples/blob/main/MultiObjectTrackingVideoFile.ipynb)|How to perform object detection with multi-object tracking (MOT) from a video file to count vehicle traffic.|
|[ObjectDetectionMultiplexingMultipleStreams](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionMultiplexingMultipleStreams.ipynb)|How to perform object detection from multiple video files, multiplexing frames. This example demonstrates lowest possible and stable AI inference latency while maintaining decent throughput. This is achieved by using synchronous prediction mode and video decoding offloaded into separate thread.|
