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
