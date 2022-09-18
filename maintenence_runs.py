import subprocess

runs_array=[0,
            0,
            0,
            0,]


result = subprocess.run(['squeue', '--me', '-o', '"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)