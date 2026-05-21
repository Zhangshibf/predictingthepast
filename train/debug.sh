#!/bin/bash
#SBATCH --job-name=inspect_released
#SBATCH --account=iscrc_coita
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/inspect_rel_%j.out
#SBATCH --error=logs/inspect_rel_%j.err

module purge
module load profile/deeplrn
source /leonardo/home/userexternal/szhang01/anaconda3/bin/activate
conda activate predictingthepast

export PYTHONUNBUFFERED=1
cd /leonardo_work/IscrC_CoIta/predictingthepast/train

python -c "
import pickle
path = '/leonardo_work/IscrC_CoIta/predictingthepast/checkpoint/aeneas_117149994_2.pkl'
with open(path, 'rb') as f:
    ckpt = pickle.load(f)
print('top-level keys:', list(ckpt.keys()))
print()
params = ckpt['params']
print('type(params):', type(params).__name__)
if hasattr(params, 'keys'):
    print('params top-level keys:', list(params.keys()))
    # Inspect one level deeper
    for k in list(params.keys())[:3]:
        v = params[k]
        if hasattr(v, 'keys'):
            print(f'  params[{k!r}] type={type(v).__name__} keys={list(v.keys())[:5]}')
        elif hasattr(v, 'shape'):
            print(f'  params[{k!r}] shape={v.shape}')
        else:
            print(f'  params[{k!r}] type={type(v).__name__}')
"