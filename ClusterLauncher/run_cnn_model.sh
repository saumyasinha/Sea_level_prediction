#!/bin/bash

# === 2. List of SBATCH arguements ===
#SBATCH --job-name=run_smallfcn_patches_changed_Validation
#SBATCH --nodelist=bgpu-dhl1
#SBATCH --account=blanca-kann
#SBATCH --gres=gpu:2
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --output=/pl/active/machinelearning/ML_for_sea_level/job_outputs/small_fcn_patches_changed_Validation.%j.out

# === 3. Purge and load needed modules ===
module purge

module load python/3.6.5

# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"

# === 5. Running the program ===
python -u ../script_learn_predictions_with_CNN.py

