import os
import subprocess
zip_arr=[]
for i in os.listdir(os.path.join('models','NMDA')):
    if 'd_r_comparison_ss_7' in i:
        zip_arr.append(i)
p = subprocess.run("zip",'-r','models_for_msc_proj.zip',*zip_arr)
