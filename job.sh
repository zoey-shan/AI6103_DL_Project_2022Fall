#!/bin/sh
#SBATCH --partition=SCSEGPU_MSAI
#SBATCH --qos=q_msai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=MyJob 
#SBATCH --output=output_%x_%j.out 
#SBATCH --error=error_%x_%j.err

module load cuda-11.7
python3 train.py --amp
