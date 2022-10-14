import subprocess
import re
import time
import sys
from train_nets.configuration_factory import change_configs_in_json ,restore_configs_from_temp,restore_optimizers_from_temp,restore_all_from_temp
models_jsons=["d_r_comparison","d_r_comparison_ss","morph","morph_linear"]
def reduce_lr(models_jsons):
    def update_lr_function(config):
        lr_value = 0.003 if "morph" in config.model_tag else 0.01
        config.constant_learning_rate=lr_value
        if 'lr' in config.optimizer_params:
            config.optimizer_params['lr']=lr_value

        # return "constant_learning_rate",config.constant_learning_rate, config.optimizer_params
    for i in models_jsons:
        change_configs_in_json(i,update_funnction=update_lr_function)
for i in models_jsons:
    restore_all_from_temp('morph')
runs_array=[
            # rf"python -c ' {rf'from maintenence_runs import reduce_lr; reduce_lr({str(models_jsons)})'} '",
            # r'python evaluation_datasets.py -j d_r_comparison -j d_r_comparison_ss -j morph  -n 15 -g False']+[
            r"python fit_CNN_execution.py d_r_comparison -g True",
            r"python fit_CNN_execution.py d_r_comparison_ss -g False -mem 120000",
            r"python fit_CNN_execution.py morph -g True",
            r"python fit_CNN_execution.py morph_linear -g True",
            ]
# for i in models_jsons:
#     restore_configs_from_temp(i)
#     restore_optimizers_from_temp(i)
# reduce_lr(models_jsons)
for i,s in enumerate(runs_array):
    print(f"Now running command: {s}")
    s=re.split(f"[\s]+",s)
    result = subprocess.run(s, stderr=subprocess.PIPE, stdout=sys.stdout)#,input=str.encode('y'))
    assert result.returncode==0 ,result.stderr
    time.sleep(1)