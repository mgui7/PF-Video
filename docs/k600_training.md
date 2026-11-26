# Kinetics-600 training

The `train_jvtc2v_k600` experiment trains class-conditioned JVT-XL/16 on
16 RGB frames at 4 fps and 256×256 resolution. Temporal grouping is 4,
giving 1,024 video tokens, and the logit-normal noise shift is 4. No padding,
gradient accumulation, VAE, or camera conditioning is used.

The 600-class model has approximately 678.15M trainable parameters. It uses
BF16 autocast, compiled transformer blocks and output layer, AdamW at 1e-4,
4,000 warmup updates, constant learning rate thereafter, and EMA 0.9999.
The current run stops at **400,000 optimizer updates**, with global batch 256.
The reference paper reports approximately one million updates. This setup does not use the
upstream scripts' dataset-dependent epoch counts as the stopping rule.

## Dataset

The default root is:

```
/e/scratch/lift-virg/fischer13/dataset/kinetics/shards-256px-4fps
```

It contains 32 training TAR shards (816,119 clips) and 8 validation TAR shards
(56,968 clips). The stored videos have 17 frames at 4 fps, with the shorter
side resized to 256px. Training selects one of the two contiguous 16-frame
windows and center-crops spatially to 256×256; the denoiser applies horizontal
flipping and class-label dropout. Sixteen frames span 3.75 seconds from the
first frame to the last.

Set `K600_ROOT` to override the root. Only `train-*.tar` files are used for
training. The loader builds a shared, locked byte-offset index in
`outputs/dataset_cache/k600-train.json` on first use; the measured initial build
took approximately two minutes. Later launches reuse it. Shards are not
extracted or modified, and 128px input is rejected rather than upscaled.

## Multi-node launch

From the repository root, with the dependencies in `requirements.txt` installed:

```bash
mkdir -p outputs
sbatch scripts/train_jvtc2v_k600_jupiter.sh
```

The Jupiter wrapper defaults to 2 nodes with 4 GPUs each (32 clips/GPU). The launcher computes
the per-GPU batch from `GLOBAL_BATCH_SIZE / (NNODES * GPUS)`:

| Nodes | GPUs/node | Clips/GPU | Global batch |
|---:|---:|---:|---:|
| 8 | 4 | 8 | 256 |
| 4 | 4 | 16 | 256 |
| 2 | 4 | 32 | 256 |

Choose a row supported by the GPU profile. For example:

```bash
sbatch --nodes=4 scripts/train_jvtc2v_k600_jupiter.sh
```

The wrapper runs one `torchrun` launcher per node through `srun`, using the
first node as `MASTER_ADDR` and each Slurm process ID as the node rank.
DistributedSampler partitions training samples across all GPU ranks. Override
`BATCH_SIZE` explicitly when a different global batch is intentional; the
launcher prints the resulting global batch. No full training job is submitted
by the setup or profiling scripts.

The default container is `/e/home/jusers/gui2/jupiter/jenv-video.sif`; override
`PF_VIDEO_CONTAINER` to use another environment. This existing container lacks
`rootutils`; the current workspace has it installed under `outputs/python`,
which the Slurm wrapper adds to `PYTHONPATH`. For a fresh checkout, install all
requirements in the environment or install that package locally:

```bash
apptainer exec --bind /e:/e /e/home/jusers/gui2/jupiter/jenv-video.sif \
  python -m pip install --no-deps --target outputs/python rootutils
```

The Jupiter normal QoS permits 12 hours per allocation. The wrapper uses an
11-hour training budget (`MAX_RUN_SECONDS=39600`), checked every 100 optimizer
updates with a shared rank-0 stop decision. It saves a final model and
optimizer/EMA/scheduler checkpoint before exiting. On successful exit below
`MAX_STEPS=400000`, it submits one dependent continuation with the same settings.
It stops submitting jobs once the checkpoint reaches the target, and does not
continue automatically after training failures or a checkpoint with no progress.
Set `AUTO_CONTINUE=0` to disable continuation submissions.

Checkpoints default to
`/e/scratch/lift-virg/gui2/PF-Video/jvtc2v_k600_16f_4fps_t4_400k/`.
Set `OUTPUT_DIR` and `NAME` to change the run directory. Continuation IDs are
appended to `continuations.tsv` there. Periodic checkpoints are also saved per
epoch. Resumption restores model, optimizer, EMA, scheduler and global step;
after a partial epoch it starts the next shuffled epoch, rather than replaying
the exact remaining sample order.

## Training capacity profile

```bash
mkdir -p outputs/profile-k600
sbatch scripts/profile_k600_jupiter.sh
```

Each candidate runs in a fresh process with single-rank DDP, FP32 parameters,
BF16 autocast, `torch.compile`, fused AdamW, GPU EMA, and four complete
forward/backward/optimizer steps. It records peak allocated/reserved GPU memory
and step times; the first two steps are excluded from the steady-state average.
Results are written to `outputs/profile-k600/<job-id>/batch-<size>.json`.

Synthetic inputs have the exact training tensor shapes. This measures model
capacity, not video-decoding throughput, multi-node communication throughput,
or checkpoint memory. Keep headroom below the largest fitting batch.

### Measured GH200 capacity (26 November 2026)

Job `3127034`, PyTorch `2.9.0a0+50eac811a6.nv25.09`, 95 GiB CUDA-visible
memory per GPU, compilation enabled. Distinct GPU UUIDs verified that concurrent
tests ran on separate physical GPUs. The earlier job `3126999` is invalid due
to shared-GPU contention and is excluded.

| Clips/GPU | Result | Peak allocated GiB | Peak reserved GiB | Steady step seconds |
|---:|---|---:|---:|---:|
| 8 | Fits | 27.44 | 28.64 | 0.124 |
| 16 | Fits | 38.11 | 39.91 | 0.205 |
| 24 | Fits | 48.81 | 50.05 | 0.289 |
| 32 | Fits | 59.48 | 61.76 | 0.394 |
| 40 | Fits | 67.65 | 70.23 | 0.465 |
| 48 | Fits | 80.82 | 82.37 | 0.567 |
| 56 | Fits, little headroom | 91.53 | 93.60 | 0.654 |
| 64 | Out of memory | — | — | — |

The largest successful tested batch is **56**, not a production recommendation.
The selected default is **32/GPU on 2 nodes × 4 GPUs = global batch 256**.
Step times average only two post-warmup iterations and exclude data loading and
cross-rank communication; they are a short capacity check, not an end-to-end
throughput benchmark. Raw records are under `outputs/profile-k600/3127034/`.

Reference: [JVT Appendix A.1](https://arxiv.org/html/2611.18549#A1.SS1).
