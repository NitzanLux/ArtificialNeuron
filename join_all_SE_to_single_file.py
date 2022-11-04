import os
import pickle
#
# for i in os.listdir('sample_entropy'):
#     cur_path =  os.path.join('sample_entropy',i)
#     if not os.path.isdir(cur_path):
#         os.remove(cur_path)
for i in os.listdir('sample_entropy'):
    cut_path =os.path.join('sample_entropy',i)
    if os.path.isdir(cut_path):
        data_dict={}
        for j in os.listdir(cut_path):
            f_path=os.path.join(cut_path,j)
            with open(f_path, 'rb') as f:
                data_dict[i]=pickle.load(f)
        with open(f'{i}.pkl','wb') as f_o:
            pickle.dump(data_dict, f_o)
