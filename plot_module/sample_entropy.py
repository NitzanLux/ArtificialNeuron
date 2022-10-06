from model_evaluation_multiple import GroundTruthData
import matplotlib.pyplot as plt
import EntropyHub as EH
import os
import numpy as np
import time
MAX_INTERVAL = 20
t = time.time()
gt_original_name = 'davids_ergodic_validation'
gt_reduction_name = 'reduction_ergodic_validation'
gt_reduction = GroundTruthData.load(os.path.join('..', 'evaluations', 'ground_truth', gt_reduction_name + '.gteval'))
gt_original = GroundTruthData.load(os.path.join('..', 'evaluations', 'ground_truth', gt_original_name + '.gteval'))

number_of_samples = -1
se_r, se_o = None, None
number_of_samples = min(number_of_samples, len(gt_reduction)) if number_of_samples>0 else len(gt_reduction)
se_r_arr = []
se_o_arr = []
cur_t=time.time()
for i, ro in enumerate(zip(gt_reduction, gt_original)):
    print(f"current sample number {i}  time: {time.time()-cur_t} seconds          total: {time.time()-t} seconds")
    cur_t=time.time()
    if i >= number_of_samples:
        break
    r, o = ro
    vr, sr = r
    vo, so = o
    se_r, _, _ = EH.SampEn(sr, m=MAX_INTERVAL)
    se_o, _, _ = EH.SampEn(so, m=MAX_INTERVAL)
    se_r_arr.append(se_r)
    se_o_arr.append(se_o)
np.save("sample_entropy_reduction.npz",np.array(se_r_arr))
np.save("sample_entropy_original.npz",np.array(se_o_arr))
t=time.time()-t
print(t)