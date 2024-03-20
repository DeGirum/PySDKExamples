
import csv
import glob
import argparse
import degirum as dg
from degirum.model import Model
import degirum_tools
from .benchmark_yaml import validate

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--key', type=str, default='', help='second model name')
    parser.add_argument('--exclude', type=str, default='some big text', help='second model name')
    parser.add_argument('--data', type=str, help='path to validation images folder')
    parser.add_argument('--annotations', type=str, help='ground truth annotation json file path')
    parser.add_argument('--cfg', type=str, default='benchmark_scripts/eval_yaml/default.yaml', help='path to eval config')
    parser.add_argument('--csv', type=str, default=None, help='second model name')
    parser.add_argument('--cloud-url', type=str, help='cloud zoo url')

    return parser.parse_args()

if __name__ == '__main__':
    args = parser_arguments()
    count = 0

    csv_file = args.csv if args.csv else f'results_{args.model}_{args.key}.csv'

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

    img_count = len(glob.glob(f"{args.data}/*.jpg"))

    machine = dg.CLOUD
    cloud_url = f"https://cs.degirum.com/{args.cloud_url}" 
    zoo = dg.connect(machine, cloud_url, degirum_tools.get_token())
    model_list = zoo.list_models()

    for model_name in model_list:
        if (args.model in model_name) and (args.key in model_name) and (args.exclude not in model_name):
            # and ('tflite_edgetpu' not in model_name):
            print('***********************************************')
            print('***********************************************')
            count += 1
            print(f'{count} = ', model_name)
            # try:
            dg_model = zoo.load_model(model_name)
            map_list = validate(dg_model=dg_model, img_folder_path=args.data, anno_json=args.annotations, cfg_yaml=args.cfg)
            
            for i, map in enumerate(map_list):
                data = [model_name, i, *map, img_count, args.data, args.annotations, args.cfg]
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
            # except Exception as e:
            #     print(f"Error in {model_name}\n", e)
            