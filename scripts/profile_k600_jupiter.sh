#!/bin/bash
#SBATCH --partition=booster
#SBATCH --account=lift-virg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --job-name=pf-k600-profile
#SBATCH --output=outputs/profile-k600/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=2
export PYTHONNOUSERSITE=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/pf-k600-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/triton"
container=${PF_VIDEO_CONTAINER:-/e/home/jusers/gui2/jupiter/jenv-video.sif}
export profile_dir="outputs/profile-k600/${SLURM_JOB_ID}"
mkdir -p "$profile_dir"
# Set visibility inside the container: its CUDA environment can override
# Slurm's per-task mask. One task owns all four allocated GPUs here.
srun --ntasks=1 --gpus-per-task=4 \
    apptainer exec --nv --bind /e:/e "$container" bash -c '
        set -euo pipefail
        for sizes in "8 16 32 48" "24 40 56 64"; do
            read -ra batches <<< "$sizes"
            pids=()
            for gpu in 0 1 2 3; do
                batch=${batches[$gpu]}
                CUDA_VISIBLE_DEVICES=$gpu \
                TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/gpu-$gpu" \
                TRITON_CACHE_DIR="$TRITON_CACHE_DIR/gpu-$gpu" \
                torchrun --standalone --nproc-per-node=1 scripts/profile_k600.py \
                    --batch-size "$batch" --output "$profile_dir/batch-${batch}.json" \
                    > "$profile_dir/batch-${batch}.log" 2>&1 &
                pids+=("$!")
            done
            failed=0
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
            if (( failed )); then exit 1; fi
        done
    '
