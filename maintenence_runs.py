import subprocess
import re
import time
import sys
runs_array=[
            "python fit_CNN_execution.py d_r_comparison -g True",
            "python fit_CNN_execution.py d_r_comparison_ss -g False -mem 120000",
            "python fit_CNN_execution.py morph -g True",
            "python fit_CNN_execution.py morph_linear -g True",
            ]+["python evaluation_dataset -j d_r_comparison -j d_r_comparison_ss -j morph -j morph_linear"]

for i,s in enumerate(runs_array):
    print(f"Now running command: {s}")
    s=re.split(f"[\s]+",s)
    result = subprocess.run(s, input='y',stderr=sys.stderr, stdout=sys.stdout)
    time.sleep(5)