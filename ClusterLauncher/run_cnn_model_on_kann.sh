#!/bin/bash
# === 2. List of SBATCH arguements ===
#SBATCH --job-name=cesm_unet_monthly_w_patches_future_data_wrmse
#SBATCH --partition=blanca-kann
#SBATCH --account=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/pl/active/machinelearning/ML_for_sea_level/job_outputs/cesm_unet_monthly_w_patches_future_data_wrmse.%j.out
# === 3. Purge and load needed modules ===
module purge
module load python/3.6.5
# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"
# === 5. Running the program ===
python -u ../script_learn_predictions_with_CNN.py
