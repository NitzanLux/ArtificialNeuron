import argparse
import os
import json
from model_evaluation import ModelEvaluator
from project_path import MODELS_DIR

parser = argparse.ArgumentParser(description='Add configuration file')
parser.add_argument(dest="configs_path", type=str,
                    help='configuration file for path')
parser.add_argument(dest="job_id", help="the job id", type=str)
args = parser.parse_args()
print(args)
with open(os.path.join(MODELS_DIR, "%s.json" % args.configs_path), 'r') as file:
    configs = json.load(file)
for i, conf in enumerate(configs):
    ModelEvaluator.build_and_save(os.path.join(MODELS_DIR, *conf))

