
import argparse
import degirum as dg
import dgtools
from pose_eval_utils import PoseModelEvaluator

cloud_token = dgtools.get_token()  # get cloud API access token from env.ini file
cloud_zoo_url = dgtools.get_cloud_zoo_url()  # get cloud zoo URL from env.ini file

# zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)
# # zoo = dg.connect('100.111.131.89', cloud_zoo_url, cloud_token)

zoo = dg.connect(dg.LOCAL, 'benchmark_scripts/yolov8n-pose/yolov8n-pose.json', cloud_token)


def validate(model_name:str, 
            img_folder_path:str, 
            anno_json:str, 
            cfg_yaml:str="benchmark_scripts/eval_yaml/default.yaml"
            ):
    model = zoo.load_model(model_name)
    
    map_evaluator = PoseModelEvaluator.init_from_yaml(
        model, cfg_yaml
    )

    map_results = map_evaluator.evaluate(
        img_folder_path,
        ground_truth_annotations_path=anno_json,
        num_val_images=10,
        print_frequency=100,
        label_check=False
    )

    return map_results

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--data', type=str, help='path to validation images folder')
    parser.add_argument('--annotations', type=str, help='ground truth annotation json file path')
    parser.add_argument('--cfg', type=str, default='benchmark_scripts/eval_yaml/default.yaml', help='path to eval config')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    
    validate(model_name=args.model, img_folder_path=args.data, anno_json=args.annotations, cfg_yaml=args.cfg)