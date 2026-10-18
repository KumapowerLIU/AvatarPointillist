import os
import time
import math
import torch
import datetime
from collections import defaultdict
from PIL import Image
import torch.nn.functional as F
import numpy as np
import logging
from avatarpointillist.argeometry.utils.misc import SmoothedValue


def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    epoch,
    args,
    filename=None,
    next_batch=0,
    global_step=0,
    scaler=None,
):
    """保存模型检查点（只在主进程调用）"""
    checkpoint_name = os.path.join(checkpoint_dir, filename)
    try:
        weight_ckpt = model.module.state_dict()
    except Exception:
        weight_ckpt = model.state_dict()

    sd = {
        "model": weight_ckpt,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "next_batch": next_batch,
        "global_step": global_step,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if scheduler is not None:
        sd["scheduler"] = scheduler.state_dict()

    torch.save(sd, checkpoint_name)
    print(f"Checkpoint saved: {checkpoint_name}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu", scaler=None):
    """加载模型检查点并恢复 optimizer 与 scheduler（若存在）"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded successfully. Epoch: {epoch}")
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return epoch, checkpoint.get("next_batch", 0), checkpoint.get("global_step", 0)


def to_rgb_image(image: Image.Image):
    """将图像转换为RGB格式"""
    if image.mode == 'RGB':
        return image
    elif image.mode == 'RGBA':
        img = Image.new("RGB", image.size, (127, 127, 127))
        img.paste(image, mask=image.getchannel('A'))
        return img
    else:
        raise ValueError(f"Unsupported image type: {image.mode}")


def do_train(
    args,
    model,
    dino_model,
    michelangelo_model,
    dataloaders,
    logger,
    accelerator

):
    """AR training with explicit optimizer-update scheduling and checkpoint metadata."""
    # 设置数据类型（用于把固定模型移到设备）
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f"Using weight dtype: {weight_dtype}")
    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda params: params.requires_grad, model.parameters()),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    # Count optimizer updates from the actual per-process loader, including partial accumulations.
    dataloaders['train'], model, optimizer = accelerator.prepare(
        dataloaders['train'], model, optimizer)
    updates_per_epoch = math.ceil(len(dataloaders['train']) / args.gradient_accumulation_steps)
    if updates_per_epoch == 0:
        raise ValueError("Training loader is empty; check dataset size, batch size, and drop_last")
    max_iters = args.max_epoch * updates_per_epoch
    if max_iters < 3:
        raise ValueError("OneCycleLR requires at least three optimizer updates")
    warmup_steps = max(2, min(max_iters - 1, args.warm_lr_epochs * updates_per_epoch))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.base_lr, total_steps=max_iters,
        pct_start=warmup_steps / max_iters, anneal_strategy='cos')
    # The scheduler is deliberately not wrapped: advance it once per successful optimizer update.
    start_epoch, next_batch, curr_iter = 0, 0, 0
    if getattr(args, 'pretrained_weights', None):
        start_epoch, next_batch, curr_iter = load_checkpoint(
            args.pretrained_weights, accelerator.unwrap_model(model),
            optimizer=None if getattr(args, 'no_load_optimizer', False) else optimizer,
            scheduler=None if getattr(args, 'no_load_optimizer', False) else scheduler,
            device='cpu', scaler=accelerator.scaler)
        if getattr(args, 'no_load_optimizer', False):
            start_epoch, next_batch, curr_iter = 0, 0, 0
    if args.start_epoch not in (-1, start_epoch):
        raise ValueError("start_epoch must match checkpoint metadata or be -1")
    args.start_epoch = start_epoch

    # 训练统计
    time_delta = SmoothedValue(window_size=10)
    model.train()

    if args.start_epoch == -1:
        args.start_epoch = start_epoch

    curr_time = time.time()
    loss_dict = defaultdict(list)

    # 将不训练的模型移到设备
    dino_model.to(accelerator.device, dtype=weight_dtype)
    dino_model.eval()

    michelangelo_model.to(accelerator.device, dtype=weight_dtype)
    michelangelo_model.eval()
    for param in michelangelo_model.parameters():
        param.requires_grad = False

    for param in dino_model.parameters():
        param.requires_grad = False

    # 主训练循环
    for curr_epoch in range(args.start_epoch, args.max_epoch):
        epoch_loss = 0.0
        num_batches = 0

        if hasattr(dataloaders['train'], 'set_epoch'):
            dataloaders['train'].set_epoch(curr_epoch)
        skipped_batches = next_batch if curr_epoch == start_epoch else 0
        epoch_loader = accelerator.skip_first_batches(dataloaders['train'], skipped_batches) if skipped_batches else dataloaders['train']
        for batch_idx, batch_data in enumerate(epoch_loader, start=skipped_batches):
            # 处理DINO特征
            with torch.no_grad():
                face_dino_imgs = batch_data["dino_imgs"]
                batch_data["face_features"] = dino_model(
                    face_dino_imgs.to(accelerator.device, dtype=weight_dtype)
                ).last_hidden_state
                batch_data["pc_feature"] = michelangelo_model.encode_latents(
                    batch_data["pc_normals"].to(accelerator.device, dtype=weight_dtype)
                )
                batch_data["shape_latents"] = michelangelo_model.to_shape_latents(batch_data["pc_feature"][:, 1:])
                batch_data['sequences'] = batch_data['sequences'].to(accelerator.device, dtype=torch.long)




            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(batch_data['sequences'], face_features=batch_data['face_features'], pc_feature=batch_data["pc_feature"] , shape_latents=batch_data["shape_latents"])
                loss = outputs

                # Every rank must take the same path before backward; skipping locally can deadlock DDP.
                finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                finite_ranks = accelerator.reduce(finite, reduction="sum")
                if finite_ranks.item() != accelerator.num_processes:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite loss on at least one training rank")

                accelerator.backward(loss)

                # 梯度裁剪
                if args.clip_gradient > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_gradient)

                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # 记录损失（保存在本进程内）
            epoch_loss += loss.item()
            num_batches += 1
            loss_dict['loss'].append(outputs.item())


            # 日志记录与检查点保存（仅在同步梯度时，并且仅主进程执行持久化相关操作）
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                time_delta.update(time.time() - curr_time)
                curr_time = time.time()
                curr_iter += 1

                # 定期日志记录
                if curr_iter % args.log_every == 0:
                    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                    eta_seconds = (max_iters - curr_iter) * time_delta.avg
                    eta_str = str(datetime.timedelta(seconds=int(max(0, eta_seconds))))

                    if accelerator.is_main_process:
                        curr_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0.0
                        logger.info(
                            f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                            f"Loss {loss.item():.6f}; "
                            f"Iter [{curr_iter}/{max_iters}]; "
                            f"LR {curr_lr:.2e}; "
                            f"Iter time {time_delta.avg:.2f}s; "
                            f"ETA {eta_str}; "
                            f"Mem {mem_mb:.2f}MB"
                        )

                        # 记录到TensorBoard（仅主进程）
                        log_dict = {
                            key: float(np.mean(value)) if len(value) > 0 else 0.0
                            for key, value in loss_dict.items()
                        }
                        log_dict["learning_rate"] = float(optimizer.param_groups[0]['lr']) if len(optimizer.param_groups) > 0 else 0.0
                        accelerator.log(log_dict, step=curr_iter)
                    loss_dict = defaultdict(list)

                # 定期保存检查点（仅主进程）
                if accelerator.is_main_process and curr_iter % args.save_every == 0:
                    save_checkpoint(
                        args.checkpoint_dir,
                        model,
                        optimizer,
                        scheduler,
                        curr_epoch,
                        args,
                        filename=f"checkpoint_{curr_iter}.pth",
                        next_batch=batch_idx + 1, global_step=curr_iter, scaler=accelerator.scaler,
                    )

        # 每个epoch结束后的日志（仅主进程输出）
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if accelerator.is_main_process:
            logger.info(f"Epoch {curr_epoch} completed. Average loss: {avg_epoch_loss:.6f}")



    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(args.checkpoint_dir, accelerator.unwrap_model(model), optimizer,
                        scheduler, args.max_epoch, args, filename="checkpoint_final.pth",
                        next_batch=0, global_step=curr_iter, scaler=accelerator.scaler)
    accelerator.end_training()
    if accelerator.is_main_process:
        logger.info("Training completed successfully!")
