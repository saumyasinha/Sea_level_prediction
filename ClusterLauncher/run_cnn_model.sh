#!/bin/bash

# === 2. List of SBATCH arguements ===
#SBATCH --job-name=cesm1and2_1yrlag_monthly_downscaled_future_data_and_Weights
#SBATCH --nodelist=bgpu-casa1
#SBATCH --account=blanca-kann
#SBATCH --gres=gpu
#SBATCH --qos=preemptable
#SBATCH --nodes=1
##SBATCH --mem-per-cpu=8g
#SBATCH --ntasks=25
#SBATCH --time=5:00:00 #1-00:00:00
#SBATCH --output=/pl/active/machinelearning/Saumya/ML_for_sea_level/job_outputs/cesm2_trend_dilatedunet_wd1e-6_0yrlag_monthly_downscaled_future_data_wrmse.%j.out


# === 3. Purge and load needed modules ===
module purge

module load python/3.6.5

# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"

# === 5. Running the program ===
#python -u ../script_learn_predictions_with_CNN.py
python -u ../script_learn_trend_or_avg_predictions_with_CNN.py
