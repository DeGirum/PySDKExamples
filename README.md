# DeGirum PySDK Examples

## Quick Start

1. Install DeGirum PySDK. Refer to [DeGirum Developers Center](https://degirum.github.io) for instructions.

1. The following script will download *MobileNetv2+SSD* CPU model from 
[DeGirum public mode zoo](https://github.com/DeGirum/public_model_zoo)
and perform ML inference of a test image with two cats. The inference result will be displayed in both text 
and graphical form.

    ```python
    import degirum as dg         # import DeGirum PySDK package
    zoo = dg.connect_model_zoo() # connect to DeGirum public model zoo
    print(zoo.list_models())     # print all available models in the model zoo

    # load mobilenet_ssd model for CPU;
    # model_name should be one returned by zoo.list_models()
    model_name = "mobilenet_v2_ssd_coco--300x300_quant_n2x_cpu_1"     
    model = zoo.load_model(model_name)

    # perform AI inference of an image specified by URL
    image_url = "https://degirum.github.io/images/samples/TwoCats.jpg"
    result = model(image_url)

    print(result)                # print numeric results
    result.image_overlay.show()  # show graphical results
    ```

## Resources

1. [DeGirum Developers Center](https://degirum.github.io): place to look for PySDK documentation
1. [DeGirum Cloud Platform](https://cs.degirum.com): place to manage your cloud API access tokens 
1. [DeGirum public mode zoo](https://github.com/DeGirum/public_model_zoo) GitHub repo

## Running PySDK Examples

This repository provides PySDK example scripts that can perform ML inferences on the following hosting options:

1. Using [DeGirum Cloud Platform](https://cs.degirum.com),
1. On DeGirum-hosted AI server node shared via Peer-to-Peer VPN,
1. On AI server node hosted by you in your local network,
1. On AI server running on your local machine,
1. On DeGirum ORCA accelerator directly installed on your local machine.

To try different options, you need to just change the `inference_option` variable in the script code.

To run the examples, clone this repo:

```
git clone https://github.com/DeGirum/PySDKExamples.git
```

Inside the repo, create an `.env` file and fill the required authentication details by assigning the following variables:

|Variable Name|Description|
|-------------|-----------|
|`DEGIRUM_CLOUD_TOKEN`|DeGirum cloud platform API access token, obtained on [DeGirum Cloud Platform](https://cs.degirum.com) site.|
|`P2P_VPN_SERVER_ADDRESS`|IP address of DeGirum-hosted AI server node shared via Peer-to-Peer VPN; please contact support@degirum.com to obtain one.|
|`LOCAL_NETWORK_SERVER_ADDRESS`|IP address of AI server node hosted by you in your local network; refer to [DeGirum Developers Center](https://degirum.github.io) for AI server installation details.|
|`GITHUB_TOKEN`|[GitHub personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) to access [DeGirum public mode zoo repo](https://github.com/DeGirum/public_model_zoo); any valid GitHub token will work.|
|`CAMERA_ID`|Local camera index or web camera URL in the format `rtsp://<user>:<password>@<ip or hostname>`|

This will allow loading the required information from the `.env` file instead of hard-coding the values in the script. 
You can copy the below lines and fill in the missing information.
```
DEGIRUM_CLOUD_TOKEN = 'Enter your DeGirum cloud platform token'
P2P_VPN_SERVER_ADDRESS = 'Enter IP address of the DeGirum P2P AI server'
LOCAL_NETWORK_SERVER_ADDRESS = 'Enter IP address of the AI server in your local network'
GITHUB_TOKEN = 'Enter your GitHub personal access token'
CAMERA_ID = 'rtsp://<user>:<password>@<ip or hostname>'
```

The `.env` file is added to `.gitignore` and will not be checked in. This will ensure that your token information is not leaked. 

## Examples Directory

| Example | Description |
|---------|-------------|
|[ObjectDetectionImage](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionImage.ipynb)|One of the most simplest examples how to do AI inference of a graphical file using object detection model.|
|[ObjectDetectionVideoFile](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionVideoFile.ipynb)|How to do AI inference of a video stream from a video file, show annotated video, and save it to another video file. |
|[ObjectDetectionCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionCameraStream.ipynb)|How to do AI inference of a video stream from a video camera and show annotated video in real-time.|
|[ObjectDetectionDataset](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionDataset.ipynb)|How to do AI inference on an image dataset and calculate performance metrics. An image dataset is retrieved from the cloud using `fiftyone` API.|
|[ObjectDetectionDatasetMultithreaded](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionDatasetMultithreaded.ipynb)|How to do **multi-threaded** AI inference on an image dataset. An image dataset is retrieved from the cloud using `fiftyone` API.|
|[ObjectDetectionVideoFile2Images](https://github.com/DeGirum/PySDKExamples/blob/main/ObjectDetectionVideoFile2Images.ipynb)|How to do AI inference of a video stream from a video file and save annotated frame images into a directory.|
|[FaceHandDetectionParallelCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/FaceHandDetectionParallelCameraStream.ipynb)|How to run two models side-by-side and combine results of both models. A video stream from a video camera is processed simultaneously by the hand and face detection models. Combined result is then displayed.|
|[PersonPoseDetectionPipelinedImage](https://github.com/DeGirum/PySDKExamples/blob/main/PersonPoseDetectionPipelinedImage.ipynb)|How to do AI inference of a graphical file using two AI models: person detection and pose detection. The person detection model is run on the image and the results are then processed by the pose detection model, one person bounding box at a time. Combined result is then displayed.|
|[PersonPoseDetectionPipelinedCameraStream](https://github.com/DeGirum/PySDKExamples/blob/main/PersonPoseDetectionPipelinedCameraStream.ipynb)|A video stream from a video camera is processed by the person detection model. The person detection results are then processed by the pose detection model, one person bounding box at a time. Combined results are then displayed as an annotated video in real-time.|
|[SoundClassificationAudioStream](https://github.com/DeGirum/PySDKExamples/blob/main/SoundClassificationAudioStream.ipynb)|How to do sound classification AI inference of an audio stream from a local microphone in real time. The result label with highest probability is displayed for each inference while keeping history few steps back.|
|[SoundClassificationAndObjectDetectionAsynchronous](https://github.com/DeGirum/PySDKExamples/blob/main/SoundClassificationAndObjectDetectionAsynchronous.ipynb)|How to perform parallel inferences on two **asynchronous** data streams with different frame rates. To achieve maximum performance this example uses **non-blocking** batch prediction mode.|
