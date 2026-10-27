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
    }
    if scheduler is not None:
        try:
            sd["scheduler"] = scheduler.state_dict()
        except Exception:
            # scheduler may not expose state_dict in some wrappers
            pass

    torch.save(sd, checkpoint_name)
    print(f"Checkpoint saved: {checkpoint_name}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    """加载模型检查点并恢复 optimizer 与 scheduler（若存在）"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            print("Warning: failed to load scheduler state")

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded successfully. Epoch: {epoch}")
    return epoch


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
    """主训练循环（使用 OneCycleLR，纯训练流程，无验证/最佳模型逻辑）

    主要变化：
    - 使用 torch.optim.lr_scheduler.OneCycleLR，并在创建后通过 accelerator.prepare 将其包含在加速器管理中。
    - checkpoint 会保存/加载 scheduler 状态。
    - 所有保存与输出（模型权重、TensorBoard 日志）仅在主进程执行。
    - 移除 best_val_metrics 与验证逻辑，仅做纯训练。
    """
    # 设置数据类型（用于把固定模型移到设备）
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f"Using weight dtype: {weight_dtype}")
    # 计算训练参数
    max_iters = args.max_epoch * len(dataloaders['train']) // max(1, args.gradient_accumulation_steps)
    if max_iters <= 0:
        max_iters = 1
    print(f"Training dataloader length: {len(dataloaders['train'])}")
    print(f"Max iterations: {max_iters}")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda params: params.requires_grad, model.parameters()),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    pct_start = 0.1
    if getattr(args, 'warm_lr_epochs', 0) and getattr(args, 'max_epoch', 1):
        pct_start = float(getattr(args, 'warm_lr_epochs', 0)) / max(1.0, float(getattr(args, 'max_epoch', 1)))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.base_lr,
        total_steps=max_iters,
        pct_start=max(0.0, min(1.0, pct_start)),
        anneal_strategy='cos',
    )
    
    # 可选加载预训练权重（注意现在不返回 best_val_metrics）
    if getattr(args, 'pretrained_weights', None) is not None:
        try:
            start_epoch = load_checkpoint(
                args.pretrained_weights,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=accelerator.device,
            )
            print(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            print("Starting training from scratch")

    # 分布式 BatchNorm 同步
    if accelerator.state.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 准备模型、优化器和数据加载器（scheduler 在准备好 max_iters 后创建并加入）
    dataloaders['train'], model, optimizer, scheduler = accelerator.prepare(
        dataloaders['train'],
        model,
        optimizer,
        scheduler
    )

    # 训练统计
    time_delta = SmoothedValue(window_size=10)
    model.train()

    if args.start_epoch == -1:
        args.start_epoch = start_epoch

    curr_iter = args.start_epoch * len(dataloaders['train']) // max(1, args.gradient_accumulation_steps)
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

        for batch_idx, batch_data in enumerate(dataloaders['train']):
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

                # 检查损失值是否合法
                if not math.isfinite(loss.item()):
                    # 仅主进程记录错误
                    if accelerator.is_main_process:
                        logger.error(f"Loss is not finite: {loss.item()}")
                    continue

                accelerator.backward(loss)

                # 梯度裁剪
                if args.clip_gradient > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_gradient)

                optimizer.step()
                # OneCycleLR expects step per optimizer.step()
                try:
                    scheduler.step()
                except Exception:
                    pass
                optimizer.zero_grad(set_to_none=True)

            # 记录损失（保存在本进程内）
            epoch_loss += loss.item()
            num_batches += 1
            loss_dict['loss'].append(outputs.item())


            # 日志记录与检查点保存（仅在同步梯度时，并且仅主进程执行持久化相关操作）
            if accelerator.sync_gradients:
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
                if accelerator.is_main_process and (curr_iter + 1) % args.save_every == 0:
                    save_checkpoint(
                        args.checkpoint_dir,
                        model,
                        optimizer,
                        scheduler,
                        curr_epoch,
                        args,
                        filename=f"checkpoint_{curr_iter+1}.pth",
                    )

        # 每个epoch结束后的日志（仅主进程输出）
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if accelerator.is_main_process:
            logger.info(f"Epoch {curr_epoch} completed. Average loss: {avg_epoch_loss:.6f}")

 

    accelerator.end_training()
    if accelerator.is_main_process:
        logger.info("Training completed successfully!")

