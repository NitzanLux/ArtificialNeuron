#!/bin/bash
# Write output as following (%j is JOB_ID)
#SBATCH -o /ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron/cluster_logs/output-%j.out
#SBATCH -e /ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron/cluster_logs/error-%j.err
# Ask for one CPU, one GPU, enter the GPU queue, and limit run to 1 days
#SBATCH -c 1
#SBATCH -t 1-0
#SBATCH -p gpu.q
#SBATCH --gres=gpu:1
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ]; then
# check the original location through scontrol and $SLURM_JOB_ID
SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
# otherwise: started with bash. Get the real location.
SCRIPT_PATH=$(realpath $0)
fi
# get script's path to allow running from any folder without errors
path=$(dirname $SCRIPT_PATH)

# put your script here - example script is sitting with this bash script
python3 $path/evaluation.py "/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/ExperimentalData/L5PC_NMDA_valid_mixed/exBas_0_1000_inhBasDiff_-800_200__exApic_0_1000_inhApicDiff_-800_200_SpTemp__saved_InputSpikes_DVTs__1566_outSpikes__128_simulationRuns__6_secDuration__randomSeed_200278.p" "models/NMDA/gaussian_train_NMDA_Tree_TCN__2021-10-10__09_32__ID_73440/gaussian_train_NMDA_Tree_TCN__2021-10-10__09_32__ID_73440.pkl" 0 90 400

