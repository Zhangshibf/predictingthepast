#!/bin/bash
#
# Launch 5 fine-tuning runs differing only in random seed, each saving a
# single checkpoint at step 10000 into its own output directory.
#
# Submit once:
#     sbatch launch_sweep.sh
#
# This creates 5 array tasks; SLURM dispatches them as GPUs become free.
# Each task writes its checkpoint to .../finetuned_seed<S>/ so the runs
# never collide. Logs land in logs/sweep_<jobid>_<seed>.{out,err}.
#
# Each task's resources are identical to launch_local.sh (1 GPU, 8 CPUs,
# 32G RAM, 10h walltime). 10000 steps at ~5 steps/sec is ~35 min of
# compute plus init overhead, so 10h is generous.

#SBATCH --job-name=paleo_sweep
#SBATCH --account=iscrc_coita
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# Seeds, in array-index order. Index $SLURM_ARRAY_TASK_ID picks one.
# Chosen as a small but arbitrary spread; change them and you change the
# experiment, so leave alone unless you're starting a new sweep.
SEEDS=(4 42 123 1337 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Per-seed output directory. config_paleo.py picks this up via the env
# var and uses it as checkpoint_dir.
export AENEAS_SEED=$SEED
export AENEAS_OUTDIR=finetuned_seed${SEED}

module purge
module load profile/deeplrn
source /leonardo/home/userexternal/szhang01/anaconda3/bin/activate
conda activate predictingthepast

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/IscrC_CoIta/predictingthepast/train

echo "---------------------------------------"
echo "Sweep task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}, outdir=${AENEAS_OUTDIR}"
echo "Time: $(date)"
echo "---------------------------------------"

python experiment.py --config=config_paleo.py --jaxline_mode=train --logtostderr

TRAIN_EXIT=$?
echo "Seed ${SEED} training finished with exit code ${TRAIN_EXIT}"