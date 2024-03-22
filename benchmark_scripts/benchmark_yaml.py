
import argparse
import degirum as dg
from degirum.model import Model
import degirum_tools
from degirum_tools.detection_eval import ObjectDetectionModelEvaluator
from degirum_tools.classification_eval import ImageClassificationModelEvaluator
from degirum_tools.regression_eval import ImageRegressionModelEvaluator


def validate(dg_model: Model,
            img_folder_path:str, 
            anno_json:str,
            task: str,
            cfg_yaml:str="benchmark_scripts/eval_yaml/default.yaml"
            ):
    
    eval_class = {
        'detect': ObjectDetectionModelEvaluator, 
        'classify': ImageClassificationModelEvaluator,
        'regress':ImageRegressionModelEvaluator
    }

    map_evaluator = eval_class[task].init_from_yaml(
        dg_model, cfg_yaml
    )

    map_results = map_evaluator.evaluate(
        image_folder_path=img_folder_path,
        ground_truth_annotations_path=anno_json,
        # print_frequency=1000,
    )

    return map_results

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--data', type=str, help='path to validation images folder')
    parser.add_argument('--task', type=str, default='detect', help='path to validation images folder')
    parser.add_argument('--annotations', type=str, help='ground truth annotation json file path')
    parser.add_argument('--cfg', type=str, default='benchmark_scripts/eval_yaml/default.yaml', help='path to eval config')
    parser.add_argument('--cloud-url', type=str, help='cloud zoo url')
    parser.add_argument('--device', type=str, default='cloud', help='device to run the model')


    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()

    device = dg.CLOUD if args.device == 'cloud' else dg.LOCAL
    cloud_url = f"https://cs.degirum.com/{args.cloud_url}" 
    zoo = dg.connect(device, cloud_url, degirum_tools.get_token())
    dg_model = zoo.load_model(args.model)
    map_results = validate(dg_model=dg_model, img_folder_path=args.data, anno_json=args.annotations, task=args.task, cfg_yaml=args.cfg)
    print(map_results)