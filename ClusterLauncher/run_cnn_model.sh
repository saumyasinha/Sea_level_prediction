#!/bin/bash

# === 2. List of SBATCH arguements ===
#SBATCH --job-name=cesm_smatunet_1yrlag_monthly_downscaled_future_data_and_Weights
#SBATCH --nodelist=bgpu-dhl1
#SBATCH --account=blanca-kann
#SBATCH --gres=gpu:2
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --output=/pl/active/machinelearning/Saumya/ML_for_sea_level/job_outputs/cesm_convlstm_more_hidden__1yr_lag_downscaled_future_data_and_weights.%j.out


# === 3. Purge and load needed modules ===
module purge

module load python/3.6.5

# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"

# === 5. Running the program ===
python -u ../script_learn_predictions_with_CNN.py

