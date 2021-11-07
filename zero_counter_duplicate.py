import os
import json
import configuration_factory as confactory
import argparse
import slurm_job
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add configuration file')
    parser.add_argument(dest="configs_paths", type=str,
                        help='configurations json file of paths')
    parser.add_argument(dest="new_json_paths", type=str,
                        help='configurations json file of paths')
    args = parser.parse_args()
    print(args)
    configs_file = args.configs_paths
    with open(os.path.join(MODELS_DIR,"%s.json"%configs_file) ,'r') as file:
        configs = json.load(file)
    new_configs=[]
    for conf in configs:
        currnt_conf = confactory.load_config_file(os.path.join(MODELS_DIR, *conf))
        currnt_conf = confactory.config_factory(save_model_to_config_dir=True,is_new_name=True,**currnt_conf)
        new_configs.append(currnt_conf)
    with open(os.path.join(MODELS_DIR,"%s.json"%args.new_json_paths), 'w') as file:
        file.write(json.dumps(new_configs))# use `json.loads` to do the reverse