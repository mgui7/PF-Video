import datetime
import numpy as np
import os
import time
import math
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler

import hydra

from omegaconf import DictConfig, OmegaConf
from typing import Union, List

import json

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.distributed_utils import (  # noqa: E402
    init_distributed_mode,
    is_main_process, is_dist_avail_and_initialized
)
from utils.checkpoint_utils import (  # noqa: E402
    save_model_safetensors, load_model_safetensors,
    save_training_state_safetensors, load_training_state_safetensors,
)
from utils.ema import EMAOptimizer  # noqa: E402
from utils.logger import MetricLogger, SmoothedValue  # noqa: E402
from utils.training_runtime import TrainingDeadline  # noqa: E402

def save_grad_norms(norms_dict, global_step, save_dir, top_k=5):
    if not is_main_process():
        return

    os.makedirs(save_dir, exist_ok=True)

    param_norms = {
        k: float(v) for k, v in norms_dict.items() 
        if "total" not in k
    }
    
    total_norm = norms_dict.get(next((k for k in norms_dict if "total" in k), None), 0.0)

    sorted_norms = sorted(param_norms.items(), key=lambda item: item[1], reverse=True)
    top_items = sorted_norms[:top_k]

    log_content = {
        "global_step": global_step,
        "total_norm": float(total_norm),
        "top_max_norms": dict(top_items),
        "all_norms": param_norms
    }

    file_path = os.path.join(save_dir, f"step_{global_step}.json")
    with open(file_path, "w") as f:
        json.dump(log_content, f, indent=4)

    print(f"\nStep: [{global_step}] Gradient Norm Summary:")
    print(f"Total Norm: {float(total_norm):.6f}")
    print(f"Top {top_k} Modules with largest Gradient Norm:")
    for name, val in top_items:
        print(f"  - {name}: {val:.6f}")


def grad_norm(module: torch.nn.Module, norm_type: Union[float, int, str], group_separator: str = "/") -> dict[str, float]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.

    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"grad_{norm_type}_norm{group_separator}{name}": p.grad.data.norm(norm_type)
        for name, p in module.named_parameters()
        if p.grad is not None
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"grad_{norm_type}_norm_total"] = total_norm
    return norms


def train_one_epoch(model, data_loader, optimizer, lr_scheduler, device, epoch, global_step, log_writer, cfg, deadline=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = max(1, int(cfg.trainer.get("print_freq", 20)))
    max_steps = int(cfg.trainer.get("max_steps", -1))
    grad_norm_freq = cfg.trainer.grad_norm_freq
    gradient_clip_val = float(cfg.trainer.get("gradient_clip_val", 0.0) or 0.0)

    pending_loss_reduce = None

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if max_steps > 0 and global_step >= max_steps:
            break

        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch["_global_step"] = global_step

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_dict = model(batch)
        loss = output_dict['loss']

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()

        # record gradient norm
        if grad_norm_freq is not None and grad_norm_freq > 0 and global_step % grad_norm_freq == 0:
            norms = grad_norm(model.module.diffusion_model, norm_type=2)
            # NOTE: `norms` need not be gathered, as they are already uniform across all devices
            save_grad_norms(
                norms_dict=norms,
                global_step=global_step,
                save_dir=os.path.join(cfg.output_dir, 'grad_norms'),
                top_k=5,
            )

        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                model.module.diffusion_model.parameters(),
                max_norm=gradient_clip_val,
                error_if_nonfinite=True,
            )

        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(loss=loss_value)
        scalar_log_values = {
            key: value
            for key, value in output_dict.items()
            if isinstance(value, (float, int))
        }
        scalar_tensor_items = []
        for key, value in output_dict.items():
            if key in {"loss", "xs", "xs_pred"}:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar_tensor_items.append((key, value.detach().float().reshape(())))
        if scalar_tensor_items:
            keys, values = zip(*scalar_tensor_items)
            values = torch.stack(values).cpu().tolist()
            scalar_log_values.update(dict(zip(keys, values)))
        metric_logger.update(**scalar_log_values)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Only fetch scalar when needed for logging
        need_log_scalar = (data_iter_step % print_freq == 0)
        if need_log_scalar and pending_loss_reduce is not None:
            loss_value_reduce_tensor, loss_reduce_handle, epoch_1000x = pending_loss_reduce
            loss_reduce_handle.wait()
            loss_value_reduce = (loss_value_reduce_tensor / cfg.infra.world_size).item()
            if log_writer is not None:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            pending_loss_reduce = None

        if need_log_scalar:
            if is_dist_avail_and_initialized() and cfg.infra.world_size > 1:
                loss_value_reduce_tensor = torch.tensor(loss_value, device=device)
                loss_reduce_handle = torch.distributed.all_reduce(loss_value_reduce_tensor, async_op=True)
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                pending_loss_reduce = (loss_value_reduce_tensor, loss_reduce_handle, epoch_1000x)
            elif log_writer is not None:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value, epoch_1000x)

            if log_writer is not None:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                for key, value in scalar_log_values.items():
                    log_writer.add_scalar(key, value, epoch_1000x)
        else:
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        global_step += 1
        if max_steps > 0 and global_step >= max_steps:
            break
        if deadline is not None and deadline.should_stop(global_step, device):
            print(f'Allocation time budget reached at update {global_step}; saving before exit.')
            break

    if pending_loss_reduce is not None:
        loss_value_reduce_tensor, loss_reduce_handle, epoch_1000x = pending_loss_reduce
        loss_reduce_handle.wait()
        loss_value_reduce = (loss_value_reduce_tensor / cfg.infra.world_size).item()
        if log_writer is not None:
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)

    return global_step


def save_model(cfg, model_without_ddp, optimizer, lr_scheduler, epoch, global_step, epoch_name=None):
    if not is_main_process():
        return

    output_dir = cfg.output_dir
    if epoch_name is None:
        epoch_name = str(epoch)
    model_path = os.path.join(output_dir, f'checkpoint_{epoch_name}.safetensors')
    metadata = {'epoch': epoch, 'global_step': global_step}
    print(f'Saving checkpoint to {model_path} ...')
    save_model_safetensors(
        model_without_ddp.state_dict(), model_path, metadata=metadata,
    )

    if epoch_name == 'last':
        # Keep the full optimizer, EMA and scheduler state for training resume.
        state_path = os.path.join(output_dir, 'checkpoint_last_training_state.safetensors')
        save_training_state_safetensors(
            optimizer.state_dict(), lr_scheduler.state_dict(), state_path,
        )
    elif isinstance(optimizer, EMAOptimizer):
        # Export EMA with model parameter names and buffers for direct loading.
        ema_path = os.path.join(output_dir, f'checkpoint_{epoch_name}_ema.safetensors')
        print(f'Saving EMA checkpoint to {ema_path} ...')
        with optimizer.swap_ema_weights():
            save_model_safetensors(
                model_without_ddp.state_dict(), ema_path, metadata=metadata,
            )

def print_cfg(cfg: DictConfig, overrides: List[str]):
    print("cfg overrides:", overrides)
    print("-" * 20 + " Configuration " + "-" * 20)
    print(OmegaConf.to_yaml(cfg))
    print("-" * 55)

    assert cfg.output_dir is not None, "Output directory must be specified."
    assert cfg.name is not None, "Experiment name must be specified."
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.name)
    os.makedirs(cfg.output_dir, exist_ok=True)

    yaml_file_suffix = os.environ.get('SLURM_JOB_ID', None)
    if yaml_file_suffix is None:
        yaml_file_suffix = time.strftime("%Y%m%d-%H%M%S")
    yaml_file = os.path.join(cfg.output_dir, f"config_{yaml_file_suffix}.yaml")
    with open(yaml_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

def main(cfg: DictConfig, overrides: List[str]):
    run_started = time.monotonic()
    # runtime infra config
    OmegaConf.set_struct(cfg, False)
    init_distributed_mode(cfg.infra)
    print_cfg(cfg, overrides)

    infra_cfg: DictConfig = cfg.infra
    trainer_cfg: DictConfig = cfg.trainer
    cfg.algorithm.global_rank = infra_cfg.rank
    cfg.algorithm.world_size = infra_cfg.world_size
    cfg.algorithm.device = infra_cfg.device

    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    # print("Arguments:\n{}".format(cfg).replace(', ', ',\n'))
    # print cfg

    device = torch.device(infra_cfg.device)

    # Set seeds for reproducibility
    seed = cfg.seed + infra_cfg.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Set up TensorBoard logging (only on main process)
    if infra_cfg.rank == 0:
        log_writer = SummaryWriter(log_dir=cfg.output_dir)
    else:
        log_writer = None

    if cfg.dataset.name == "kinetics600":
        from dataset.kinetics600 import Kinetics600VideoDataset
        dataset_train = Kinetics600VideoDataset(cfg=cfg.dataset, current_epoch=0)
    elif cfg.dataset.name == "realestate10k":
        from dataset.re10k import RealEstate10KVideoDataset
        dataset_train = RealEstate10KVideoDataset(cfg=cfg.dataset, current_epoch=0)
    elif cfg.dataset.name == "ucf101":
        from dataset.ucf101 import UCF101VideoDataset
        dataset_train = UCF101VideoDataset(cfg=cfg.dataset, current_epoch=0)
    elif cfg.dataset.name in ("faceforensics", "faceforensics-train"):
        from dataset.uncond import FaceForensicsVideoDataset
        dataset_train = FaceForensicsVideoDataset(cfg=cfg.dataset, current_epoch=0)
    elif cfg.dataset.name in ("skytimelapse", "skytimelapse-train"):
        from dataset.uncond import SkyTimelapseVideoDataset
        dataset_train = SkyTimelapseVideoDataset(cfg=cfg.dataset, current_epoch=0)
    elif cfg.dataset.name in ("taichi_hd", "taichi-train"):
        from dataset.uncond import TaichiHDVideoDataset
        dataset_train = TaichiHDVideoDataset(cfg=cfg.dataset, current_epoch=0)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=infra_cfg.world_size, rank=infra_cfg.rank, 
        shuffle=True, seed=cfg.seed,
    )
    print("Sampler_train =", sampler_train)

    data_loader_kwargs = {
        "dataset": dataset_train,
        "sampler": sampler_train,
        "batch_size": trainer_cfg.data.batch_size,
        "shuffle": False,
        "persistent_workers": (trainer_cfg.data.num_workers > 0),
        "num_workers": trainer_cfg.data.num_workers,
        "pin_memory": trainer_cfg.data.pin_mem,
        "drop_last": True,
    }
    if trainer_cfg.data.num_workers > 0 and trainer_cfg.data.prefetch_factor is not None:
        data_loader_kwargs["prefetch_factor"] = trainer_cfg.data.prefetch_factor

    data_loader_train = torch.utils.data.DataLoader(**data_loader_kwargs)
    print(f"Dataset contains {len(dataset_train):,} videos")

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    if cfg.algorithm.name == "camctrl_denoiser":
        from jvt.camctrl_denoiser import CamCtrlDenoiser
        model = CamCtrlDenoiser(cfg.algorithm)
    elif cfg.algorithm.name == "dit_c2video":
        from jvt.c2v_denoiser import C2VDenoiser
        model = C2VDenoiser(cfg.algorithm)
    elif cfg.algorithm.name == "dit_video_uncond":
        from jvt.uncond_denoiser import UnCondDenoiser
        model = UnCondDenoiser(cfg.algorithm)
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")
    # model.to(device)  ## done inside model
    print("Model =", model)

    # for batch in data_loader_train:
    #     model.sample_all_videos(batch)

    ddp_kwargs = {
        "module": model,
        "device_ids": [infra_cfg.gpu],
        "find_unused_parameters": trainer_cfg.find_unused_parameters,
        "gradient_as_bucket_view": trainer_cfg.gradient_as_bucket_view,
        "static_graph": trainer_cfg.static_graph,
    }
    if trainer_cfg.bucket_cap_mb is not None:
        ddp_kwargs["bucket_cap_mb"] = trainer_cfg.bucket_cap_mb

    model = torch.nn.parallel.DistributedDataParallel(**ddp_kwargs)
    model_without_ddp = model.module

    n_params = sum(p.numel() for p in model_without_ddp.diffusion_model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    # Set up optimizer with weight decay adjustment for bias and norm layers
    transition_params = list(model_without_ddp.diffusion_model.parameters())
    base_optimizer = torch.optim.AdamW(transition_params, **trainer_cfg.optimizer)
    print("Base optimizer =", base_optimizer)

    lr_scheduler = get_scheduler(optimizer=base_optimizer, **trainer_cfg.lr_scheduler)
    print("LR scheduler =", lr_scheduler)

    if trainer_cfg.ema.enable:
        ema_cfg = trainer_cfg.ema
        optimizer = EMAOptimizer(
            base_optimizer,
            device=device,
            decay=ema_cfg.decay,
            every_n_steps=ema_cfg.every_n_steps,
            current_step=0,
        )
        print("Using EMA optimizer with decay =", ema_cfg.decay)
    else:
        optimizer = base_optimizer

    # Resume from checkpoint if provided
    checkpoint_path = trainer_cfg.resume
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print("Resuming checkpoint from", trainer_cfg.resume)

        model_state, meta = load_model_safetensors(checkpoint_path)
        model_without_ddp.load_state_dict(model_state)
        start_epoch = int(meta.get('epoch', -1)) + 1
        global_step = int(meta.get('global_step', start_epoch * len(data_loader_train)))
        del model_state
        # Load training state from companion safetensors
        state_path = checkpoint_path.replace('.safetensors', '_training_state.safetensors')
        if os.path.exists(state_path):
            opt_state, sched_state = load_training_state_safetensors(state_path)
            optimizer.load_state_dict(opt_state)
            lr_scheduler.load_state_dict(sched_state)
            del opt_state, sched_state
        else:
            print(f"Warning: training state file {state_path} not found, only loading model weights.")

        print(f"Loaded optimizer & scaler state! Start training from epoch {start_epoch}, global_step {global_step}")
    else:
        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            print(f"Warning: checkpoint path {checkpoint_path} not exists, train from scratch...")
        start_epoch = 0
        global_step = 0
        print("Training from scratch")

    eff_batch_size = trainer_cfg.data.batch_size * infra_cfg.world_size
    print("Learning rate: {:.2e}".format(trainer_cfg.optimizer.lr))
    print("Effective batch size: %d" % eff_batch_size)
    if float(trainer_cfg.get("gradient_clip_val", 0.0) or 0.0) > 0:
        print(f"Gradient clipping max norm: {trainer_cfg.gradient_clip_val}")

    # Training loop
    print(f"Start training for {trainer_cfg.max_epoches} epochs")
    start_time = time.time()
    deadline = TrainingDeadline(float(trainer_cfg.get('max_run_seconds', 0)), started_at=run_started)

    for epoch in range(start_epoch, trainer_cfg.max_epoches):
        if infra_cfg.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        dataset_train.set_epoch(epoch)
        global_step = train_one_epoch(model, data_loader_train, optimizer, lr_scheduler, device, epoch, global_step, log_writer, cfg, deadline)
        reached_target = trainer_cfg.max_steps > 0 and global_step >= trainer_cfg.max_steps

        # Save checkpoint periodically
        if not bool(trainer_cfg.get("disable_checkpoint", False)):
            if deadline.stopped or reached_target or epoch % trainer_cfg.save_last_freq == 0 or epoch + 1 == trainer_cfg.max_epoches:
                save_model(
                    cfg=cfg, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    epoch=epoch, global_step=global_step, epoch_name="last"
                )

            if epoch % trainer_cfg.save_freq == 0 and epoch > 0:
                save_model(
                    cfg=cfg, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    epoch=epoch, global_step=global_step
                )
        
        # barrier
        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

        # Perform online evaluation at specified intervals
        # if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
        #     torch.cuda.empty_cache()
        #     with torch.no_grad():
        #         evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
        #     torch.cuda.empty_cache()

        if log_writer is not None:
            log_writer.flush()

        if trainer_cfg.max_steps > 0 and global_step >= trainer_cfg.max_steps:
            print(f"Reached max_steps={trainer_cfg.max_steps}; stopping training.")
            break
        if deadline.stopped:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    overrides = sys.argv[1:]

    with hydra.initialize(version_base="1.2", config_path="configs"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    main(cfg, overrides)
