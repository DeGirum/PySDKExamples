import degirum as dg
import dgtools
from dgtools.detection_eval import ObjectDetectionModelEvaluator

cloud_token = dgtools.get_token()  # get cloud API access token from env.ini file
cloud_zoo_url = dgtools.get_cloud_zoo_url()  # get cloud zoo URL from env.ini file

zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)

model_name = "yolo_v5s_coco--512x512_quant_n2x_orca_1"

model = zoo.load_model(model_name)

map_evaluator = ObjectDetectionModelEvaluator.init_from_yaml(
    model, "benchmark_scripts/eval_yaml/coco.yaml"
)

img_folder_path = "/data/ml-data/datasets/coco/images/val2017"
anno_json = "/data/ml-data/datasets/coco/annotations/instances_val2017.json"

map_results = map_evaluator.evaluate(
    img_folder_path,
    ground_truth_annotations_path=anno_json,
    num_val_images=0,
    print_frequency=100,
)
