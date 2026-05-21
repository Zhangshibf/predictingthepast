#!/bin/bash
#SBATCH --job-name=inspect_ckpt
#SBATCH --account=iscrc_coita
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/inspect_%j.out
#SBATCH --error=logs/inspect_%j.err

module purge
module load profile/deeplrn
source /leonardo/home/userexternal/szhang01/anaconda3/bin/activate
conda activate predictingthepast

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/IscrC_CoIta/predictingthepast/train

CHECKPOINT="/leonardo_work/IscrC_CoIta/predictingthepast/finetuned_seed4/checkpoint_latest_id0000.pkl"

python -c "
import pickle, jax

with open('${CHECKPOINT}', 'rb') as f:
    snapshot_list = pickle.load(f)
snap = snapshot_list[-1]
nest = snap.pickle_nest
if hasattr(nest, 'to_dict'):
    nest = nest.to_dict()
exp_state = nest['experiment_module']
if hasattr(exp_state, 'to_dict'):
    exp_state = exp_state.to_dict()
params = exp_state.get('_params') or exp_state.get('params')

print('=== Top-level structure of params ===')
print(type(params))
if hasattr(params, 'keys'):
    print('keys:', list(params.keys())[:5])

print()
print('=== Shape of every leaf, grouped by path ===')
def walk(p, prefix=''):
    if hasattr(p, 'shape'):
        print(f'{prefix:60s} shape={p.shape} dtype={p.dtype}')
    elif hasattr(p, 'keys'):
        for k, v in p.items():
            walk(v, f'{prefix}/{k}')
walk(params)

print()
print('jax.local_device_count() =', jax.local_device_count())
"