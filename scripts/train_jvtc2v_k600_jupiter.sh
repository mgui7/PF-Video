#!/bin/bash
#SBATCH --partition=booster
#SBATCH --account=lift-virg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --job-name=pf-k600
#SBATCH --output=outputs/k600-slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NNODES=$SLURM_NNODES
export GPUS=4
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
export MAX_STEPS=${MAX_STEPS:-400000}
export MAX_RUN_SECONDS=${MAX_RUN_SECONDS:-39600}
export AUTO_CONTINUE=${AUTO_CONTINUE:-1}
export OUTPUT_DIR=${OUTPUT_DIR:-/e/scratch/lift-virg/gui2/PF-Video}
export NAME=${NAME:-jvtc2v_k600_16f_4fps_t4_400k}
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/outputs/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/pf-k600-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/triton"
container=${PF_VIDEO_CONTAINER:-/e/home/jusers/gui2/jupiter/jenv-video.sif}
checkpoint="$OUTPUT_DIR/$NAME/checkpoint_last.safetensors"
read_step() {
    apptainer exec --bind /e:/e "$container" python -c '
import sys
from pathlib import Path
from safetensors import safe_open
path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    with safe_open(str(path), framework="pt") as f:
        print(int(f.metadata()["global_step"]))
' "$checkpoint"
}
start_step=$(read_step)
if (( start_step >= MAX_STEPS )); then
    echo "Target already reached: $start_step / $MAX_STEPS updates"
    exit 0
fi
if [[ -f "$checkpoint" && ! -f "${checkpoint%.safetensors}_training_state.safetensors" ]]; then
    echo 'Resume checkpoint is missing its optimizer/EMA state; stopping.' >&2
    exit 1
fi
srun --ntasks="$NNODES" --ntasks-per-node=1 --gpus-per-task=4 \
    apptainer exec --nv --bind /e:/e "$container" \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jvtc2v_k600.sh "$@" \
    trainer.max_steps="$MAX_STEPS" trainer.lr_scheduler.num_training_steps="$MAX_STEPS" \
    trainer.max_run_seconds="$MAX_RUN_SECONDS"
end_step=$(read_step)
if (( end_step <= start_step )); then
    echo "No checkpoint progress ($start_step -> $end_step); not submitting a continuation." >&2
    exit 1
fi
if (( end_step < MAX_STEPS && AUTO_CONTINUE )); then
    next_job=$(sbatch --parsable --nodes="$NNODES" \
        --dependency="afterok:$SLURM_JOB_ID" --job-name="pf-k600-${MAX_STEPS}" \
        scripts/train_jvtc2v_k600_jupiter.sh "$@")
    echo "Saved update $end_step / $MAX_STEPS; continuation job: $next_job"
    printf '%s\t%s\t%s\n' "$SLURM_JOB_ID" "$end_step" "$next_job" >> "$OUTPUT_DIR/$NAME/continuations.tsv"
else
    echo "Saved update $end_step / $MAX_STEPS; no continuation submitted."
fi
