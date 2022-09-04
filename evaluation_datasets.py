from model_evaluation_multiple import *
from utils.slurm_job import *
from project_path import *
if __name__ == '__main__':
    job_factory = SlurmJobFactory("cluster_logs")
    for i in ['reduction_comparison_expend_1___2022-08-25__14_16__ID_57604'
              ,'reduction_comparison_expend_1_reduction___2022-08-25__14_16__ID_31311'
              ,'reduction_comparison_expend_3___2022-08-25__14_16__ID_47104',
              'reduction_comparison_expend_3_reduction___2022-08-25__14_16__ID_56711',
                'reduction_comparison_expend_5_reduction___2022-08-25__14_16__ID_29885'
              ,'reduction_comparison_expend_5___2022-08-25__14_16__ID_37019'
              ,'reduction_comparison_expend_7_reduction___2022-08-25__14_16__ID_6065'
              ,'reduction_comparison_expend_7___2022-08-25__14_16__ID_1939']:
        if 'reduction___'in i:
            gt_name= 'reduction_ergodic_validation'
        else:
            gt_name= 'davids_ergodic_validation'
        job_factory.send_job('model_%s'%gt_name,'python -c "from model_evaluation_multiple import create_model_evaluation;'
                                                ' create_model_evaluation(%s,%s)"'%("'" + gt_name + "'", "'" + i + "'")
                             , run_on_GPU=True)
    # run_test()