#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export K600_ROOT=${K600_ROOT:-/e/scratch/lift-virg/fischer13/dataset/kinetics/shards-256px-4fps}
GPUS=${GPUS:-4}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_PROCID:-0}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
if (( GLOBAL_BATCH_SIZE % (NNODES * GPUS) != 0 )); then
    echo 'GLOBAL_BATCH_SIZE must be divisible by NNODES * GPUS' >&2
    exit 1
fi
BATCH_SIZE=${BATCH_SIZE:-$(( GLOBAL_BATCH_SIZE / (NNODES * GPUS) ))}
NUM_WORKERS=${NUM_WORKERS:-4}
NAME=${NAME:-jvtc2v_k600_16f_4fps_t4}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
RESUME=${RESUME:-$OUTPUT_DIR/$NAME/checkpoint_last.safetensors}

if (( NNODES == 1 )); then
    distributed_args=(--standalone --nnodes=1)
else
    : "${MASTER_ADDR:?Set MASTER_ADDR to the first node hostname for multi-node training}"
    distributed_args=(--nnodes="$NNODES" --node-rank="$NODE_RANK" \
        --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29500}")
fi
echo "Training: $NNODES nodes x $GPUS GPUs x $BATCH_SIZE clips = $((NNODES * GPUS * BATCH_SIZE)) global batch"
torchrun "${distributed_args[@]}" --nproc_per_node="$GPUS" \
    train.py experiment=train_jvtc2v_k600 \
    name="$NAME" output_dir="$OUTPUT_DIR" \
    trainer.data.batch_size="$BATCH_SIZE" \
    trainer.num_nodes="$NNODES" \
    trainer.data.num_workers="$NUM_WORKERS" \
    trainer.resume="$RESUME" "$@"
