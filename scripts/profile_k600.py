"""Profile full JVT training steps, including DDP, AdamW state and GPU EMA.

Run each batch size in a fresh process. Synthetic clips isolate model capacity
from storage/decoding performance; this does not measure end-to-end throughput.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from jvt.c2v_denoiser import C2VDenoiser
from utils.ema import EMAOptimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    if args.steps < 3:
        parser.error('Use at least three steps to initialize optimizer state and DDP buckets.')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    dist.init_process_group('nccl')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    if args.eager:
        torch._dynamo.config.disable = True
    with hydra.initialize_config_dir(version_base='1.2', config_dir=str(Path(__file__).resolve().parents[1] / 'configs')):
        cfg = hydra.compose(config_name='config', overrides=['experiment=train_jvtc2v_k600'])
    OmegaConf.set_struct(cfg, False)
    cfg.algorithm.global_rank = dist.get_rank()
    cfg.algorithm.world_size = dist.get_world_size()
    result = {
        'batch_size_per_gpu': args.batch_size, 'gpu': torch.cuda.get_device_name(),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_uuid': str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
        'total_memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
        'torch': torch.__version__, 'compiled': not args.eager,
        'frames': cfg.dataset.max_frames, 'resolution': cfg.dataset.resolution,
        'temporal_factor': cfg.algorithm.temporal_downsampling_factor,
        'shift': cfg.algorithm.noise_sampling.shift,
        'world_size': dist.get_world_size(), 'precision': 'bf16 mixed',
        'optimizer': 'fused AdamW', 'ema': 0.9999, 'activation_checkpointing': False,
        'accumulation': 1,
    }
    start = time.perf_counter()
    try:
        denoiser = C2VDenoiser(cfg.algorithm)
        result['trainable_parameters'] = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        model = torch.nn.parallel.DistributedDataParallel(
            denoiser, device_ids=[torch.cuda.current_device()],
            gradient_as_bucket_view=True, static_graph=cfg.trainer.static_graph,
        )
        optimizer = EMAOptimizer(
            torch.optim.AdamW(denoiser.diffusion_model.parameters(), **cfg.trainer.optimizer),
            device=torch.device('cuda'), decay=cfg.trainer.ema.decay,
        )
        batch = {
            'videos': torch.rand(args.batch_size, cfg.dataset.max_frames, *cfg.algorithm.x_shape, device='cuda'),
            'conds': torch.randint(600, (args.batch_size,), device='cuda'),
        }
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for step in range(args.steps):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch)
            loss = output['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            loss_value = loss.item()
            optimizer.step()
            optimizer.join()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - step_start)
            print(json.dumps({'step': step, 'loss': loss_value, 'seconds': durations[-1],
                              'peak_gib': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
            del output, loss
        result.update(status='ok', step_seconds=durations,
                      steady_step_seconds=sum(durations[2:]) / len(durations[2:]))
    except torch.OutOfMemoryError as exc:
        result.update(status='oom', error=str(exc))
    except Exception as exc:
        result.update(status='error', error=str(exc))
        raise
    finally:
        result.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      elapsed_seconds=time.perf_counter() - start)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
