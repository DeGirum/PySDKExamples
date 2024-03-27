import argparse
import degirum as dg
import degirum_tools
from degirum_tools.regression_eval import ImageRegressionModelEvaluator
import csv
import json

cloud_token = degirum_tools.get_token()  # get cloud API access token from env.ini file
cloud_zoo_url = degirum_tools.get_cloud_zoo_url()  # get cloud zoo URL from env.ini file

zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)
model_list = zoo.list_models()

def validate(model_name:str, 
            img_count:int,
            img_folder_path:str, 
            anno_json:str, 
            cfg_yaml:str="benchmark_scripts/eval_yaml/regress.yaml"
            ):
    model = zoo.load_model(model_name)
    regress_evaluator = ImageRegressionModelEvaluator.init_from_yaml(
        model, cfg_yaml
    )
    
    regress_results = regress_evaluator.evaluate(
        img_folder_path,
        ground_truth_annotations_path=anno_json,
        max_images=img_count
    )

    return regress_results

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--key', type=str, default='', help='second model name')
    parser.add_argument('--exclude', type=str, default='some big text', help='second model name')
    parser.add_argument('--data', type=str, help='path to validation images folder')
    parser.add_argument('--annotations', type=str, help='ground truth annotation json file path')
    parser.add_argument('--cfg', type=str, default='benchmark_scripts/eval_yaml/regress.yaml', help='path to eval config')
    parser.add_argument('--csv', type=str, default=None, help='results CSV file name')

    return parser.parse_args()

if __name__ == '__main__':
    args = parser_arguments()
    count = 0

    csv_file = args.csv if args.csv else f'results_{args.model}_{args.key}.csv'

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

    img_count = 0

    for model_name in model_list:
        if (args.model in model_name) and (args.key in model_name) and (args.exclude not in model_name):
            print('***********************************************')
            print('***********************************************')
            count += 1
            print(f'{count} = ', model_name)
            try:
                regress_list = validate(model_name=model_name, img_count=img_count, img_folder_path=args.data, anno_json=args.annotations, cfg_yaml=args.cfg)
                
                if img_count == 0:
                    with open(args.annotations, "r") as fi:
                        anno = json.load(fi)
                        img_count = len(anno["images"])
                
                for i, metrics in enumerate(regress_list):
                    data = [model_name, i, *metrics, img_count, args.data, args.annotations, args.cfg]
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(data)
            except Exception as e:
                print(f"Error in {model_name}\n", e)