#!/bin/bash
#SBATCH --job-name=aeneas_smoke
#SBATCH --account=iscrc_coita
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

module purge
module load profile/deeplrn
source /leonardo/home/userexternal/szhang01/anaconda3/bin/activate
conda activate predictingthepast

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PROJECT_ROOT="/leonardo_work/IscrC_CoIta/predictingthepast"
WINDOWS_PATH='/leonardo_work/IscrC_CoIta/predictingthepast/aeneas_test_windows.json'
DAMAGES_PATH="/leonardo_work/IscrC_CoIta/predictingthepast/damage_spans_aeneas.json"
CONFIG_PATH="${PROJECT_ROOT}/train/config_paleo_eval.py"

cd "${PROJECT_ROOT}/train"

SEEDS=(4)

for SEED in "${SEEDS[@]}"; do
    CHECKPOINT_PATH="${PROJECT_ROOT}/finetuned_seed${SEED}/checkpoint_flat.pkl"

    echo
    echo "========================================="
    echo "SMOKE TEST: seed=${SEED}"
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    echo "Time: $(date)"
    echo "========================================="

    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        echo "  MISSING checkpoint, skipping."
        continue
    fi

    python evaluate_finetuned.py \
        --checkpoint "${CHECKPOINT_PATH}" \
        --windows "${WINDOWS_PATH}" \
        --damages "${DAMAGES_PATH}" \
        --setting both \
        --max-windows 2 \
        --topn 10\
        --out-prefix "smoke_seed${SEED}"

    SEED_EXIT=$?
    echo "seed=${SEED} exit code: ${SEED_EXIT}"
done

echo
echo "All seeds done."