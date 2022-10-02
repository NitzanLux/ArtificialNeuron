import os
import  shutil
from project_path import *

if __name__ == '__main__':
    for i in os.listdir(MODELS_DIR):
        if not os.path.isdir(i):
            continue
        best_arr=[]
        creat_folder_flag=True
        for f in os.listdir(i):
            if 'best' in f and 'temp' in f:
                print(f)

                best_arr.append(f)
            elif 'best' in f and 'temp' not in f:
                creat_folder_flag=False
        if len(best_arr)==0:
            continue
        dest=best_arr[0].replace('temp','')
        if creat_folder_flag:
            os.path.mkdir(os.path.join(MODELS_DIR,i,dest))
        for f in best_arr:
            print(dest)
            shutil.move(os.path.join(MODELS_DIR,i,f),os.path.join(MODELS_DIR,i,dest))