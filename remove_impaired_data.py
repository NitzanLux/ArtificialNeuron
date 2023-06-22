import os
import shutil

base_parent_dir = "/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/data/L5PC_Hay_Learning/"
for i in ['train','validation','test']:
    # parent directory where your directories (samples) are stored

    parent_dir=base_parent_dir+i+"/"
    # get a list of all directories
    all_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # count number of files in each directory and track the maximum number
    max_files = 0
    for directory in all_dirs:
        dir_path = os.path.join(parent_dir, directory)
        num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        if num_files > max_files:
            max_files = num_files

    # iterate over all directories again
    for directory in all_dirs:
        dir_path = os.path.join(parent_dir, directory)
        num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

        # if the directory has fewer files than the maximum, remove it
        if num_files < max_files:
            shutil.rmtree(dir_path)
