from PIL import Image
import cv2
import numpy as np

def resize(im, smaller_size):
    """
    Resize image bilinearly to make shorter side at least as long as the dimension provided.
    """
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = cv2.resize(im, (int(h_res), int(w_res)), interpolation=cv2.INTER_LINEAR)
    else:
        im_res = im
    return im_res

def preprocess(im: Image, normalization: dict, window_size: int) -> np.ndarray:
    """
    Preprocess image: scale, normalize, unsqueeze, and resize

    :param im: input image
    :param normalization: dictionary containing normalization data
    :return:
            im: processed (scaled and normalized) image
    """
    # change PIL image to NumPy array and scale to [0, 1]
    im = np.asarray(im, dtype=np.float32) / 255.0
    # normalize by given mean and standard deviation
    im -= np.array(normalization["mean"])[None, None, :]
    im /= np.array(normalization["std"])[None, None, :]
    # HWC -> CHW
    im = np.transpose(im, (2, 0, 1))
    # change dim from [C, H, W] to [1, C, H, W]
    im = np.expand_dims(im, axis=0)
    # resize image to window size by shorter dimension
    im = resize(im, window_size)

    return im