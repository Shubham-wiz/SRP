#!/usr/bin/env bash
#SBATCH --job-name=final_labour
#SBATCH --output=final_labour%j.log
#SBATCH --error=final_labour%j.err
#SBATCH --mail-user=jaffery@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:3
#SBATCH --mail-type=FAIL
srun python3 experiments/run_experiment_with_best_hps.py --dataset labour --method DeepVAR
