# DeGirum PySDK Examples

## PySDK Installation

It is recommended to install DeGirum PySDK package inside a virtual environment. Currently, Python 3.8 and 3.9 are supported. Other versions will be supported in future.

To install DeGirum PySDK from the DeGirum index server use the following command:

```
python -m pip install degirum --extra-index-url https://degirum.github.io/simple
```

To force reinstall the most recent PySDK version without reinstalling all dependencies use the following command:
```
python -m pip install degirum --upgrade --no-deps --force-reinstall --extra-index-url https://degirum.github.io/simple
```

## Resources

[DeGirum Developers Center](https://degirum.github.io)

## Quick Start

```python
# import DeGirum PySDK package
import degirum as dg

# connect to DeGirum public model zoo
zoo = dg.connect_model_zoo()

# load mobilenet_ssd model for CPU
mobilenet_ssd = zoo.load_model("mobilenet_v2_ssd_coco--300x300_quant_n2x_cpu_1")

# perform AI inference of an image specified by URL
image_url="https://degirum.github.io/images/samples/TwoCats.jpg"
result = mobilenet_ssd(image_url)

# print numeric results
print(result)

# show graphical results
result.image_overlay.show()
```

## Running PySDK Examples

This repository prvides some example scripts that can work with 

0. DeGirum Cloud Server, 
1. AI server equipped with DeGirum ORCA accelerator shared via Peer-to-Peer VPN, 
2. AI server equipped with DeGirum ORCA accelerator running in local network and 
3. AI server equipped with DeGirum ORCA accelerator running on the same machine as this code. 

To try different options, the user needs to just change the model zoo option in the code.

To run the examples, clone this repo:

```
git clone https://github.com/DeGirum/PySDKExamples.git
```

Inside the repo, create an .env file and fill the required information such as DEGIRUM_CLOUD_TOKEN, P2P_VPN_SERVER_ADDRESS etc. This will allow loading the required information from the .env file instead of hardcoding the values in the script. You can copy the below lines and fill in the missing information.
```
DEGIRUM_CLOUD_SERVER_ADDRESS='dgcps://cs.degirum.com'

DEGIRUM_CLOUD_TOKEN='Enter your token here'

P2P_VPN_SERVER_ADDRESS='Enter the IP address of the P2P_VPN_SERVER'

LOCAL_NETWORK_SERVER_ADDRESS='Enter the IP address of the AI server running in local network'

LOCAL_HOST_ADDRESS='localhost'
```
The .env file is added to .gitignore and will not be checked in. This will ensure that your token information is not leaked. 


