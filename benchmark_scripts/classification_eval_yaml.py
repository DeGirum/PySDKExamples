
import argparse
import degirum as dg
import degirum_tools
from degirum_tools.classification_eval import ImageClassificationModelEvaluator

cloud_token = degirum_tools.get_token()  # get cloud API access token from env.ini file
cloud_zoo_url = degirum_tools.get_cloud_zoo_url()  # get cloud zoo URL from env.ini file

zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)

def validate(model_name:str, 
            img_folder_path:str, 
            cfg_yaml:str="benchmark_scripts/eval_yaml/default-cls.yaml"
            ):
    model = zoo.load_model(model_name)
    map_evaluator = ImageClassificationModelEvaluator.init_from_yaml(
        model, cfg_yaml
    )
    accuracy, accuracy_per_cls = map_evaluator.evaluate(
        img_folder_path,
    )

    return accuracy, accuracy_per_cls

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--data', type=str, help='path to validation images folder')
    parser.add_argument('--cfg', type=str, default='benchmark_scripts/eval_yaml/default.yaml', help='path to eval config')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    
    validate(model_name=args.model, img_folder_path=args.data, cfg_yaml=args.cfg)