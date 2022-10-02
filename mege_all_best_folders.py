import os
import  shutil
from project_path import *

if __name__ == '__main__':
    for i in os.listdir(MODELS_DIR):
        if not os.path.isdir(os.path.join(MODELS_DIR,i)):
            continue
        best_arr=[]
        creat_folder_flag=True

        for f in os.listdir(os.path.join(MODELS_DIR,i)):
            if 'best' in f:
                print(f)
                for k in os.listdir(os.path.join(MODELS_DIR,i,f)):
                    cur_path = os.path.join(MODELS_DIR,i,f,k)
                    if os.path.isdir(cur_path):
                        for h in os.listdir(cur_path):
                            shutil.move(os.path.join(cur_path,h),os.path.join(MODELS_DIR, i, f))
                        if len(os.listdir(cur_path))==0:
                            os.remove(cur_path)
        # for f in os.listdir(os.path.join(MODELS_DIR,i)):
        #     if 'best' in f and 'temp' in f:
        #         print(f)
        #
        #         best_arr.append(f)
        #     elif 'best' in f and 'temp' not in f:
        #         creat_folder_flag=False
        # if len(best_arr)==0:
        #     continue
        # dest=best_arr[0].replace('temp','')
        # if creat_folder_flag:
        #     os.path.mkdir(os.path.join(MODELS_DIR,i,dest))
        # for f in best_arr:
        #     print(dest)
        #     shutil.move(os.path.join(MODELS_DIR,i,f),os.path.join(MODELS_DIR,i,dest))