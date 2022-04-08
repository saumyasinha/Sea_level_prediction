#!/bin/bash
# === 2. List of SBATCH arguements ===
#SBATCH --job-name=cesm1_and2_combined_monthly_downscaled_future_data_wrmse
#SBATCH --partition=blanca-kann
#SBATCH --account=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --mem=256g
#SBATCH --time=5:00:00 #1-00:00:00
#SBATCH --output=/pl/active/machinelearning/Saumya/ML_for_sea_level/job_outputs/cesm1_and2_combined_unet_attn_monthly_downscaled_future_data_wrmse.%j.out
# === 3. Purge and load needed modules ===
module purge
module load python/3.6.5
# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"
# === 5. Running the program ===
python -u ../script_learn_predictions_with_CNN.py
#python -u ../script_learn_mapping_from_clm_to_obs.py
