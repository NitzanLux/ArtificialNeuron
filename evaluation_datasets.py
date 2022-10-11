from model_evaluation_multiple import *
from utils.slurm_job import *
from project_path import *
from utils.general_aid_function import get_works_on_cluster
import platform
import re
import argparse
import logging
if __name__ == '__main__':
    if platform.system() == 'Windows':
        run_test(False)
    else:
        parser = argparse.ArgumentParser(description='json files')

        parser.add_argument('-j', dest="json_files_name",action='append', type=str, nargs='+', help='jsons files names')
        parser.add_argument('-re', dest="json_regex_query", type=str, help='jsons files regex query',default='.*')
        parser.add_argument('-n', dest="jobs_number",type=int, help='number of jobs to use',default=-1)
        parser.add_argument('-g', dest="use_gpu", type=str,
                            help='true if to use gpu false otherwise', default="False")

        args = parser.parse_args()
        use_gpu = not args.use_gpu.lower() in {"false", '0', ''}
        m_query=re.compile(f"{args.json_regex_query}")

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
            if not m_query.match(i):
                continue
            if i.endswith('.config'):
                i=i[:-len('.config')]
                print(i)
            if 'reduction___'in i:
                gt_name= 'reduction_ergodic_validation'
            else:
                gt_name= 'davids_ergodic_validation'
            commands.append('python -c "from model_evaluation_multiple import create_model_evaluation;'
                                                    ' create_model_evaluation(%s,%s)"'%("'" + gt_name + "'", "'" + i + "'") )
        assert len(commands)>0, "no files that match regex pattern or exists in the json"
        number_of_jobs=min(number_of_jobs, len(commands))
        jumps= len(commands)//number_of_jobs
        if len(get_works_on_cluster("model_evaluations_[0-9]+"))>0:
            logging.warning("evaluation is already in progress, please wait")
            exit(0)
        for c,i in enumerate(range(0, len(commands), jumps)):
            command=" && ".join(commands[i:min(i+jumps,len(commands))])
            keys={}
            if not use_gpu:
                keys={'mem':120000}
            job_factory.send_job(f'model_evaluations_{c}',command, run_on_GPU=use_gpu,**keys)
