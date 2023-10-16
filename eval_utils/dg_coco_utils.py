import numpy as np

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def save_results_coco_json(results, jdict, image_id, class_map=None):
    max_category_id = 0
    for result in results:
        box = xyxy2xywh(np.asarray(result['bbox']).reshape(1,4)*1.0)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy ce
        box=box.reshape(-1).tolist()
        category_id=class_map[result['category_id']] if class_map else result['category_id']
        jdict.append({'image_id': image_id,
                      'category_id': category_id,
                      'bbox': [np.round(x, 3) for x in box],
                      'score': np.round(result['score'], 5)})
        max_category_id = max(max_category_id, category_id)
    return max_category_id