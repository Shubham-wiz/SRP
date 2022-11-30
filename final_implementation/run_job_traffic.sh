#!/usr/bin/env bash
#SBATCH --job-name=final_traffic
#SBATCH --output=final_traffic%j.log
#SBATCH --error=final_traffic%j.err
#SBATCH --mail-user=jaffery@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
srun python3 experiments/run_experiment_with_best_hps.py --dataset traffic --method DeepVAR
