import os
import shutil
import numpy as np

# parent directory where your directories (samples) are stored
parent_dir = "/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/neuron_mi/neuron_mi/simulations/data/dendritic_tree_data_long/"

# directories to store training, validation, and testing sets
base_dir = "/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/data/L5PC_Hay_Learning_long"  # base directory where train, test, val folders will be created
# import os
# import shutil
# import numpy as np

# parent directory where your directories (samples) are stored
# parent_dir = "/path/to/your/parent/directory/"

# directories to store training, validation, and testing sets
# base_dir = "/path/to/base/directory/"  # base directory where train, test, val folders will be created
train_dir = os.path.join(base_dir, "train/")
val_dir = os.path.join(base_dir, "validation/")
test_dir = os.path.join(base_dir, "test/")

# get a list of all directories
all_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

# count number of files in each directory and track the maximum number
max_files = 0
for directory in all_dirs:
    dir_path = os.path.join(parent_dir, directory)
    num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    if num_files > max_files:
        max_files = num_files

# filter out directories with fewer than the maximum number of files
max_dirs = [d for d in all_dirs if len([f for f in os.listdir(os.path.join(parent_dir, d))
                                         if os.path.isfile(os.path.join(parent_dir, d, f))]) == max_files]

# print directories for confirmation
for d in max_dirs:
    print(d)

# get confirmation from user
confirmation = input("Are all the directories to move okay? (yes/no): ")

if confirmation.lower() != 'yes':
    print("Operation cancelled.")
else:
    # randomly shuffle the directories
    np.random.shuffle(max_dirs)

    # calculate split indices
    train_split_idx = int(len(max_dirs) * 0.7)
    val_split_idx = int(len(max_dirs) * 0.85)

    # split directories
    train_dirs = max_dirs[:train_split_idx]
    val_dirs = max_dirs[train_split_idx:val_split_idx]
    test_dirs = max_dirs[val_split_idx:]

    # create directories if not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # function to move directories
    def move_dirs(dirs, new_location):
        for d in dirs:
            shutil.move(os.path.join(parent_dir, d), os.path.join(new_location, d))

    # move directories
    move_dirs(train_dirs, train_dir)
    move_dirs(val_dirs, val_dir)
    move_dirs(test_dirs, test_dir)

    print("Operation completed.")

# train_dir = os.path.join(base_dir, "train/")
# val_dir = os.path.join(base_dir, "validation/")
# test_dir = os.path.join(base_dir, "test/")
#
# # get a list of all directories
# all_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
#
# # randomly shuffle the directories
# np.random.shuffle(all_dirs)
#
# # calculate split indices
# train_split_idx = int(len(all_dirs)*0.7)
# val_split_idx = int(len(all_dirs)*0.85)
#
# # split directories
# train_dirs = all_dirs[:train_split_idx]
# val_dirs = all_dirs[train_split_idx:val_split_idx]
# test_dirs = all_dirs[val_split_idx:]
#
# # create directories if not exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
#
# # function to move directories
# def move_dirs(dirs, new_location):
#     for d in dirs:
#         shutil.move(os.path.join(parent_dir, d), os.path.join(new_location, d))
#
# # move directories
# move_dirs(train_dirs, train_dir)
# move_dirs(val_dirs, val_dir)
# move_dirs(test_dirs, test_dir)
