import os
import subprocess
zip_arr=[]
for i in os.listdir(os.path.join('models','NMDA')):
    if 'd_r_comparison_ss_7' in i:
        zip_arr.append(os.path.join('models','NMDA',i,i+'_best'))
p = subprocess.Popen(["zip",'-r','models_for_msc_proj.zip',*zip_arr])
