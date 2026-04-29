#!/bin/bash



#SBATCH --job-name=paleo_train
#SBATCH --account=iscrc_coita            # Use lowercase as seen in saldo/sshare
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                     # Changed from bprod to normal
#SBATCH --time=10:00:00                  # Increased time for your 20 epochs
#SBATCH --nodes=1                        # You only need 1 node
#SBATCH --ntasks=1                       # 1 task for a single loop
#SBATCH --cpus-per-task=8                # 8 CPUs is the standard 'slice' for 1 GPU
#SBATCH --gres=gpu:1                     # Requesting 1 GPU
#SBATCH --mem=32G                        # 32G is plenty for 1 GPU
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module purge
module load profile/deeplrn
source /leonardo/home/userexternal/szhang01/anaconda3/bin/activate
conda activate predictingthepast

# Optional: Set environment variables for performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Set OpenMP threads per task
export NCCL_DEBUG=INFO                        # Enable NCCL debugging (for multi-GPU communication)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/IscrC_CoIta/predictingthepast/train

echo "---------------------------------------"
echo "Starting the job"
echo "Time: $(date)"
echo "---------------------------------------"

python experiment.py --config=config_paleo.py --jaxline_mode=train --logtostderr

TRAIN_EXIT=$?
echo "Training finished with exit code $TRAIN_EXIT"