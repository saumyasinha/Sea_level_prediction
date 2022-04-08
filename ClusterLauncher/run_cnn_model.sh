#!/bin/bash

# === 2. List of SBATCH arguements ===
#SBATCH --job-name=cesm1and2_1yrlag_monthly_downscaled_future_data_and_Weights
#SBATCH --nodelist=bgpu-mktg1
#SBATCH --account=blanca-kann
#SBATCH --gres=gpu
#SBATCH --qos=preemptable
#SBATCH --nodes=1
##SBATCH --mem-per-cpu=8g
#SBATCH --ntasks=50
#SBATCH --output=/pl/active/machinelearning/Saumya/ML_for_sea_level/job_outputs/cesm1and2_unet_1yr_lag_downscaled_future_data_and_weights.%j.out


# === 3. Purge and load needed modules ===
module purge

module load python/3.6.5

# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"

# === 5. Running the program ===
python -u ../script_learn_predictions_with_CNN.py

