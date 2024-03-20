import argparse
import degirum as dg
import degirum_tools
from degirum_tools.regression_eval import ImageRegressionModelEvaluator

cloud_token = degirum_tools.get_token()  # get cloud API access token from env.ini file
cloud_zoo_url = degirum_tools.get_cloud_zoo_url()  # get cloud zoo URL from env.ini file

zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)

def validate(model_name:str, 
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
        num_val_images=0,
        print_frequency=1000,
    )

    return regress_results

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--data', type=str, help='path to validation images folder')
    parser.add_argument('--annotations', type=str, help='ground truth annotation json file path')
    parser.add_argument('--cfg', type=str, default='benchmark_scripts/eval_yaml/regress.yaml', help='path to eval config')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    
    validate(model_name=args.model, img_folder_path=args.data, anno_json=args.annotations, cfg_yaml=args.cfg)