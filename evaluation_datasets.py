from model_evaluation_multiple import *
from utils.slurm_job import *
from project_path import *
import platform
import argparse

if __name__ == '__main__':
    if platform.system() == 'Windows':
        run_test()
    else:
        parser = argparse.ArgumentParser(description='json files')

        parser.add_argument('-j', dest="json_files_name",action='append', type=str, nargs='+', help='jsons files names')
        parser.add_argument('-n', dest="jobs_number",type=int, help='number of jobs to use',default=-1)
        parser.add_argument('-g', dest="use_gpu", type=str,
                            help='true if to use gpu false otherwise', default="False")

        args = parser.parse_args()
        use_gpu = not args.use_gpu.lower() in {"false", '0', ''}


        job_factory = SlurmJobFactory("cluster_logs")
        configs_lists=[]
        print(args)
        for json_name in args.json_files_name:
            if isinstance(json_name,list):
                json_name=json_name[0]
            with open(os.path.join(MODELS_DIR, "%s.json" % json_name), 'r') as file:
                configs_lists.extend(json.load(file))
        commands=[]
        number_of_jobs = args.jobs_number

        for i in configs_lists:
            i=i[1]
            if i.endswith('.config'):
                i=i[:-len('.config')]
                print(i)
            if 'reduction___'in i:
                gt_name= 'reduction_ergodic_validation'
            else:
                gt_name= 'davids_ergodic_validation'
            commands.append('python -c "from model_evaluation_multiple import create_model_evaluation;'
                                                    ' create_model_evaluation(%s,%s)"'%("'" + gt_name + "'", "'" + i + "'") )
        number_of_jobs=min(number_of_jobs, len(commands))
        jumps= len(commands)//number_of_jobs
        for c,i in enumerate(range(0, len(commands), jumps)):
            command=" && ".join(commands[i:min(i+jumps,len(commands))])
            keys={}
            if not use_gpu:
                keys={'mem':120000}
            job_factory.send_job(f'model_evaluations_{c}',command, run_on_GPU=use_gpu,**keys)

    # for i in ["morph_7___2022-09-07__23_01__ID_42876",
    #             "morph_7_reduction___2022-09-07__23_01__ID_28654",
    #             "d_r_comparison_7___2022-09-07__22_59__ID_57875",
    #             "morph_linear_7___2022-09-15__17_55__ID_30822",
    #             "d_r_comparison_7_reduction___2022-09-07__22_59__ID_31437",
    #             "d_r_comparison_5___2022-09-07__22_59__ID_65381",
    #             "d_r_comparison_5_reduction___2022-09-07__22_59__ID_9020",
    #             "d_r_comparison_3___2022-09-07__22_59__ID_53410",
    #             "d_r_comparison_3_reduction___2022-09-07__22_59__ID_44648",
    #             "d_r_comparison_1___2022-09-07__22_59__ID_14835",
    #             "d_r_comparison_1_reduction___2022-09-07__22_59__ID_14945"]:
