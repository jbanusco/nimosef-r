#!/bin/bash
#SBATCH --job-name=nimosef_val
#SBATCH --output=/path/to/logs/nimosef_val_%A_%a.out
#SBATCH --error=/path/to/logs/nimosef_val_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --qos=16cpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --partition=rad2
#SBATCH --account=rad
#SBATCH --exclude=gpunode01

# Singularity folder (edit to your local/cluster path)
singularity_path=/path/to/SingularityImages
singularity_img=${singularity_path}/nimosef_0.0.sif

# Dataset path (edit to your local/cluster path)
dataset_path=/path/to/ACDC/MIDS/mixed

# Code path (edit to your local/cluster path)
code_path='/path/to/nimosef-r'

# Logs path
logs_folder=${dataset_path}/derivatives/nimosef_logs
mkdir -p ${logs_folder}

# Docker mapping
docker_data='/usr/data'
docker_code='/usr/src'
docker_log='/usr/logs'

# Path to config file
config_file="${code_path}/nimosef/config/config_baseline_validation.json"

# Extract parameters from JSON using Python
config_params=$(python3 -c "
import json
with open('${config_file}', 'r') as f:
    config = json.load(f)
# Convert JSON keys and values to command-line arguments
print(' '.join(f'--{key} {value}' for key, value in config.items() if value is not None))
")
echo "Using config parameters: ${config_params}"

# Run inference in the dataset
singularity exec --nv \
--bind ${dataset_path}:${docker_data} \
--bind ${code_path}:${docker_code} \
--bind ${logs_folder}:${docker_log} \
${singularity_img} /bin/bash -c "cd ${docker_code} && python3 -m nimosef.training.inference ${config_params}"
