#!/usr/bin/env bash
#SBATCH --job-name=final_wiki
#SBATCH --output=final_wiki%j.log
#SBATCH --error=final_wiki%j.err
#SBATCH --mail-user=jaffery@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
srun python3 experiments/run_experiment_with_best_hps.py --dataset wiki --method DeepVAR
