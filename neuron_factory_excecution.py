import slurm_job
import argparse
import os

parser = argparse.ArgumentParser(description='Simulate NEURON and send multiple jobs')

parser.add_argument(dest="environment_name", type=str,
                    help='environment name for simulation')
parser.add_argument(dest="number of simulations", type=int,
                    help='number of simulations to run')

parser.add_argument(dest="total_sim_duration_in_sec", type=int,
                    help='simulation duration in seconds')

parser.add_argument(dest="collect_and_save_DVTs", type=bool,
                    help='is dendritic voltage is needed', default=False)

args = parser.parse_args()


print(args)

job_factory = SlurmJobFactory("neuron simulations")

for i in range(args.number_of_simulations):
    job_factory.send_job("%i_%s_simulation" % (i, args.environment_name),
                         'python3 $(dirname "$path")/NEURON_models_maker/neuron_factory.py %s %i %r $SLURM_JOB_ID' %
                         (args.environment_name,args.total_sim_duration_in_sec,args.collect_and_save_DVTs), True)