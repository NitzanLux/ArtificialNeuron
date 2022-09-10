from model_evaluation_multiple import *
from utils.slurm_job import *
from project_path import *
if __name__ == '__main__':
    job_factory = SlurmJobFactory("cluster_logs")

    for i in ["morph_7___2022-09-07__23_01__ID_42876",
                "morph_7_reduction___2022-09-07__23_01__ID_28654",
                "d_r_comparison_7___2022-09-07__22_59__ID_57875",
                "d_r_comparison_7_reduction___2022-09-07__22_59__ID_31437",
                "d_r_comparison_5___2022-09-07__22_59__ID_65381",
                "d_r_comparison_5_reduction___2022-09-07__22_59__ID_9020",
                "d_r_comparison_3___2022-09-07__22_59__ID_53410",
                "d_r_comparison_3_reduction___2022-09-07__22_59__ID_44648",
                "d_r_comparison_1___2022-09-07__22_59__ID_14835",
                "d_r_comparison_1_reduction___2022-09-07__22_59__ID_14945"]:
        if 'reduction___'in i:
            gt_name= 'reduction_ergodic_validation'
        else:
            gt_name= 'davids_ergodic_validation'
        job_factory.send_job('model_%s'%gt_name,'python -c "from model_evaluation_multiple import create_model_evaluation;'
                                                ' create_model_evaluation(%s,%s)"'%("'" + gt_name + "'", "'" + i + "'"))
    #                          , run_on_GPU=True)
    # run_test()