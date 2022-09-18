import subprocess

runs_array=[
            "python fit_CNN_execution.py d_r_comparison -g True",
            "python fit_CNN_execution.py d_r_comparison_ss -g False -mem 120000",
            "python fit_CNN_execution.py morph -g True",
            "python fit_CNN_execution.py morph_linear -g True",
            "python evaluation_dataset"]


result = subprocess.run(['squeue', '--me', '-o', '"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)