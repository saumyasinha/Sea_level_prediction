#!/bin/bash
# === 2. List of SBATCH arguements ===
#SBATCH --job-name=FCN_first_Attempt
#SBATCH --partition=blanca-kann
#SBATCH --account=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=1-00:00:00
#SBATCH --output=/pl/active/machinelearning/ML_for_sea_level/job_outputs/FCN_first_Attempt.%j.out
# === 3. Purge and load needed modules ===
module purge
module load python/3.6.5
# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"
# === 5. Running the program ===
python -u ../script_learn_predictions_with_CNN.py
