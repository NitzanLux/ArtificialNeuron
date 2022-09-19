import subprocess
import re
import time
import sys
from train_nets.configuration_factory import change_configs_in_json
models_jsons=["d_r_comparison","d_r_comparison_ss","morph","morph_linear"]
def reduce_lr(models_jsons):
    def update_lr_function(config):
        config.constant_learning_rate=config.constant_learning_rate/2
        if 'lr' in config.optimizer_params:
            config.optimizer_params['lr']=config.optimizer_params['lr']/2
        return config.constant_learning_rate, config.optimizer_params
    for i in models_jsons:
        change_configs_in_json(i,update_funnction=update_lr_function)
runs_array=[
            f"python -c ' {f'from maintenence_runs import reduce_lr; reduce_lr({str(models_jsons)})'} '",
            "python evaluation_datasets.py -j d_r_comparison -j d_r_comparison_ss -j morph -j morph_linear -n 1 -g True"]+[
            "python fit_CNN_execution.py d_r_comparison -g True",
            "python fit_CNN_execution.py d_r_comparison_ss -g False -mem 120000",
            "python fit_CNN_execution.py morph -g True",
            "python fit_CNN_execution.py morph_linear -g True",
            ]

for i,s in enumerate(runs_array):
    print(f"Now running command: {s}")
    s=re.split(f"[\s]+",s)
    result = subprocess.run(s, input=str.encode('y'),stderr=subprocess.PIPE, stdout=sys.stdout)
    assert result is None ,result
    time.sleep(1)