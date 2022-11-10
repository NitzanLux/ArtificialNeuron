import os
import subprocess
zip_arr=[]
for i in os.listdir(os.path.join('models','NMDA')):
    if i=="comparison_3__reduction___2022-10-19__15_39__ID_75531" or i =="comparison_3____2022-10-19__15_39__ID_46379":
        zip_arr.append(os.path.join('models','NMDA',i,i+'_best'))
p = subprocess.Popen(["zip",'-r','models_for_msc_proj.zip',*zip_arr])

