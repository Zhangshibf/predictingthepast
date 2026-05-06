#!/bin/bash

#SBATCH --job-name=paleo_eval
#SBATCH --account=iscrc_coita
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00                  # eval is fast; 1h is plenty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

module purge
module load profile/deeplrn
source /leonardo/home/userexternal/szhang01/anaconda3/bin/activate
conda activate predictingthepast

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/IscrC_CoIta/predictingthepast/train

echo "---------------------------------------"
echo "Starting eval on dev set"
echo "Time: $(date)"
echo "---------------------------------------"

python experiment.py --config=config_paleo_eval.py \
    --jaxline_mode=eval \
    --logtostderr

EVAL_EXIT=$?
echo "Eval finished with exit code $EVAL_EXIT"