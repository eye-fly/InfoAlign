#!/bin/bash
#SBATCH --account=hai_1275     # Run 'jsc-projects' to find your active project ID
#SBATCH --partition=booster          # Target the Booster (GPU) module
#SBATCH --nodes=1                    # Request 1 compute node
#SBATCH --ntasks-per-node=4          # 4 tasks (typically matching 4 GPUs per node)
#SBATCH --gres=gpu:4                 # Request all 4 GPUs on the node
#SBATCH --time=05:00:00              # Wall clock time limit (HH:MM:SS)
#SBATCH --output=job_output.%j.txt   # Standard output log (%j inserts the job ID)
#SBATCH --error=job_error.%j.txt    # Standard error log

# 1. Clean and load the correct environment modules
module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load Python/3.13.5
module load CUDA

# 2. Activate your virtual environment if you have one
source .venv/bin/activate

# 3. Execute your program (using srun inside the script for parallel tasks)
srun python train.py --pretrain-dataset mayaanlab --dataset finetune-chembl2k --with-ge-decoder --with-cp-decoder --pretrain-epochs 100 --finetune-epochs 60