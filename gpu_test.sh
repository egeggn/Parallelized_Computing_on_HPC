#!/bin/bash
#SBATCH --job-name=jax_gpu_test
#SBATCH --partition=gpu_a100     # Adjust to your cluster's partition name
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=4        # Adjust based on your needs
#SBATCH --mem=16G                # Adjust based on your needs
#SBATCH --time=01:00:00          # Set a reasonable limit
#SBATCH --output=%j.out

# Clear all inherited cluster modules to prevent library pollution
module purge
module load anaconda/2024-10 # load anaconda

# Activate your conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jax_gpu_env

# Tell JAX exactly where to find ptxas and nvvm within your conda env
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

# Prevent JAX from pre-allocating 90% of VRAM immediately
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run test code
echo "Starting GPU test programm..."

# Run the verification script 
python jax_cuda_minimal_test.py
