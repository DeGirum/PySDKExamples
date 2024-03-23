
import csv
import glob
import argparse
import degirum as dg
import degirum_tools
from degirum.model import Model
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
    parser.add_argument('--model', type=str, nargs='+', default=[], help='in model name')
    parser.add_argument('--exclude', type=str, nargs='+', default=[], help='not in model name')
    parser.add_argument('--data', type=str, help='path to validation images folder')
    parser.add_argument('--annotations', type=str, help='ground truth annotation json file path')
    parser.add_argument('--cfg', type=str, default='benchmark_scripts/eval_yaml/default.yaml', help='path to eval config')
    parser.add_argument('--csv', type=str, default=None, help='second model name')
    parser.add_argument('--cloud-url', type=str, help='cloud zoo url')
    parser.add_argument('--device', type=str, default='cloud', help='device to run the model')
    parser.add_argument('--task', type=str, default='detect', help='model task: detect, classify, regress, pose etc.')
    

    return parser.parse_args()

if __name__ == '__main__':
    args = parser_arguments()
    count = 0

    default_csv = 'results_' + '_'.join(args.model) + "_excl_" + '_'.join(args.exclude) + '.csv'

    csv_file = args.csv if args.csv else default_csv

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

    img_count = len(glob.glob(f"{args.data}/*.jpg"))

    device = dg.CLOUD if args.device == 'cloud' else dg.LOCAL
    cloud_url = f"https://cs.degirum.com/{args.cloud_url}" 
    zoo = dg.connect(device, cloud_url, degirum_tools.get_token())
    model_list = zoo.list_models()

    for model_name in model_list:

        if all(model_key in model_name for model_key in args.model) and all(model_key not in model_name for model_key in args.exclude):
            # and ('tflite_edgetpu' not in model_name):
            print('***********************************************')
            print('***********************************************')
            count += 1
            print(f'{count} = ', model_name)
            # try:
            dg_model = zoo.load_model(model_name)
            map_list = validate(dg_model=dg_model, img_folder_path=args.data, anno_json=args.annotations, task=args.task, cfg_yaml=args.cfg)
            
            for i, map in enumerate(map_list):
                data = [model_name, i, *map, img_count, args.data, args.annotations, args.cfg]
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
            # except Exception as e:
            #     print(f"Error in {model_name}\n", e)
            