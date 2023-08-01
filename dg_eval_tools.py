import requests
import json
import tempfile
import os
import numpy as np
from contextlib import contextmanager

@contextmanager
def annotations_accessor(url:str):
    """Annotation accessor context manager: downloads annotations into local temporary file and returns its name
    
    - `url`: annotations url 
    """
    r = requests.get(url)
    r.raise_for_status()
    fd, fname = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(r.json(), f)
        yield fname
    finally:
        os.unlink(fname)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def save_results_coco_json(results, jdict, image_id, class_map):
    for result in results:
        box = xyxy2xywh(np.asarray(result['bbox']).reshape(1,4)*1.0)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy ce
        box=box.reshape(-1).tolist()
        jdict.append({'image_id': image_id,
                      'category_id': class_map[result['category_id']],
                      'bbox': [np.round(x, 3) for x in box],
                      'score': np.round(result['score'], 5)})