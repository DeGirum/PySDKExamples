import degirum as dg
import dgtools
from dgtools.detection_eval import ObjectDetectionModelEvaluator

cloud_token = dgtools.get_token()  # get cloud API access token from env.ini file
cloud_zoo_url = dgtools.get_cloud_zoo_url()  # get cloud zoo URL from env.ini file

zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)

model_name = "yolo_v5s_coco--512x512_quant_n2x_orca_1"

model = zoo.load_model(model_name)

classmap = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]

map_evaluator = ObjectDetectionModelEvaluator(model, classmap=classmap)

img_folder_path = "/data/ml-data/datasets/coco/images/val2017"
anno_json = "/data/ml-data/datasets/coco/annotations/instances_val2017.json"

map_results = map_evaluator.evaluate(
    img_folder_path,
    ground_truth_annotations_path=anno_json,
    num_val_images=0,
    print_frequency=100,
)
