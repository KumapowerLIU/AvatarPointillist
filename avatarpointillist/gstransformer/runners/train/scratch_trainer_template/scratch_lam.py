# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import gc
import math
from tqdm.auto import tqdm
import torch
import logging
import argparse
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from torchvision.utils import save_image
from torch.utils.data import DataLoader
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed

# from .base_trainer import Trainer
from .scratch_base_trainer import BasicTrainer
from avatarpointillist.gstransformer.utils.seeds import set_seeds
from avatarpointillist.gstransformer.utils.profiler import DummyProfiler
from avatarpointillist.gstransformer.runners import REGISTRY_RUNNERS
from avatarpointillist.gstransformer.datasets import MixerDataset, DynamicDataManager


torch.multiprocessing.set_sharing_strategy('file_system')


# logger = get_logger(__name__)


@REGISTRY_RUNNERS.register('train.lam')
class LAMTrainer(BasicTrainer):
    def __init__(self):
        super().__init__()
        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.train_data_manager, self.val_data_manager = self._build_dataloader(self.cfg)
        self.train_loader, self.val_loader = self.train_data_manager.get_loader(), self.val_data_manager.get_loader()
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self.pixel_loss_fn, self.perceptual_loss_fn, self.tv_loss_fn, self.offset_reg_loss_fn = self._build_loss_fn(self.cfg)
        self.stop_train_trigger = False

    def _build_model(self, cfg):
        assert cfg.experiment.type == 'lam', \
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from avatarpointillist.gstransformer.models import ModelLAM
        model = ModelLAM(**cfg.model)
        return model

    def _build_optimizer(self, model: nn.Module, cfg):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        logging.info("======== Weight Decay Parameters ========")
        logging.info(f"Total: {len(decay_params)}")
        logging.info("======== No Weight Decay Parameters ========")
        logging.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )

        return optimizer

    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.world_size)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logging.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logging.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        if cfg.train.scheduler.type == 'cosine':
            from avatarpointillist.gstransformer.utils.scheduler import CosineWarmupScheduler
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler

    def _build_dataloader(self, cfg):
        # build dataset
        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.data.subsets,
            sample_side_views=cfg.data.dataset.sample_side_views,
            source_image_res=cfg.data.dataset.source_image_res,
            render_image_res_low=cfg.data.dataset.render_image.low,
            render_image_res_high=cfg.data.dataset.render_image.high,
            render_region_size=cfg.data.dataset.render_image.region,
            repeat_num=cfg.data.dataset.repeat_num,
            aspect_standard=cfg.data.dataset.aspect_standard,
            enlarge_ratio=cfg.data.dataset.enlarge_ratio,
        )
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.data.subsets,
            sample_side_views=cfg.data.dataset.sample_side_views,
            source_image_res=cfg.data.dataset.source_image_res,
            render_image_res_low=cfg.data.dataset.render_image.low,
            render_image_res_high=cfg.data.dataset.render_image.high,
            render_region_size=cfg.data.dataset.render_image.region,
            repeat_num=cfg.data.dataset.repeat_num,
            aspect_standard=cfg.data.dataset.aspect_standard,
            enlarge_ratio=cfg.data.dataset.enlarge_ratio,
        )

        # build data manager
        train_data_manager = DynamicDataManager(
            dataset=train_dataset,
            max_img_per_gpu=cfg.data.loader.max_img_per_gpu,
            image_num=cfg.data.loader.image_num,
            num_workers=cfg.data.loader.num_workers.train,
            pin_memory=cfg.data.loader.pin_memory,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )
        val_data_manager = DynamicDataManager(
            dataset=val_dataset,
            max_img_per_gpu=cfg.data.loader.max_img_per_gpu,
            image_num=cfg.data.loader.image_num,
            num_workers=cfg.data.loader.num_workers.val,
            pin_memory=cfg.data.loader.pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
        )

        return train_data_manager, val_data_manager

    def _build_loss_fn(self, cfg):
        from avatarpointillist.gstransformer.losses import PixelLoss, LPIPSLoss, TVLoss, OffsetReg
        pixel_loss_fn = PixelLoss()
        # with self.accelerator.main_process_first():
        # if self.is_main_process:
        perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)
        tv_loss_fn = TVLoss()
        offset_reg_loss_fn = OffsetReg(sigma=cfg.train.loss.offset_sigma)
        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn, offset_reg_loss_fn

    def register_hooks(self):
        """Register any necessary hooks"""
        pass

    def forward_loss_local_step(self, data):
        source_c2ws = data['source_c2ws']
        source_intrs = data['source_intrs']
        render_c2ws = data['c2ws']
        render_intrs = data['intrs']

        source_image = data['source_rgbs']
        render_image = data['render_image']
        render_mask = data['render_mask']
        render_full_resolutions = data['render_full_resolutions']
        render_bg_colors = data['render_bg_colors']

        flame_params = dict(
            expr=data['expr'],
            rotation=data['rotation'],
            neck_pose=data['neck_pose'],
            jaw_pose=data['jaw_pose'],
            eyes_pose=data['eyes_pose'],
            translation=data['translation'],
            betas=data['betas'],
        )

        N, M, C, H, W = render_image.shape

        # forward
        outputs = self.model(
            image=source_image,
            source_c2ws=source_c2ws,
            source_intrs=source_intrs,
            render_c2ws=render_c2ws,
            render_intrs=render_intrs,
            render_bg_colors=render_bg_colors,
            flame_params=flame_params,

            # render_resolutions=render_full_resolutions,
            # render_region_size=self.cfg.dataset.render_image.region,
        )

        cano_gs_lst = outputs['cano_gs_lst']
        offset_list = [_gs.offset.unsqueeze(0) for _gs in cano_gs_lst]
        offsets = torch.cat(offset_list, dim=0)

        # loss calculation
        loss = 0.
        loss_pixel = None
        loss_perceptual = None
        # loss_tv = None
        loss_offset_reg = None

        # if self.cfg.train.loss.pixel_weight > 0.:
        #     loss_pixel = self.pixel_loss_fn(outputs['images_rgb'], render_image)
        #     loss += loss_pixel * self.cfg.train.loss.pixel_weight

        # device = outputs['comp_rgb'].device
        # render_image = render_image.to(device)
        # render_mask = render_mask.to(device)

        if self.cfg.train.loss.masked_pixel_weight > 0.:
            loss_pixel = self.pixel_loss_fn(outputs['comp_rgb'], render_image)
            loss += loss_pixel * self.cfg.train.loss.masked_pixel_weight

        if self.cfg.train.loss.perceptual_weight > 0.:
            loss_perceptual = self.perceptual_loss_fn(outputs['comp_rgb'], render_image)
            loss += loss_perceptual * self.cfg.train.loss.perceptual_weight

        # if self.cfg.train.loss.tv_weight > 0.:
        #     loss_tv = self.tv_loss_fn(outputs['planes'])
        #     loss += loss_tv * self.cfg.train.loss.tv_weight

        if self.cfg.train.loss.offset_reg_weight > 0.:
            loss_offset_reg = self.offset_reg_loss_fn(offsets)
            loss += loss_offset_reg * self.cfg.train.loss.offset_reg_weight

        return outputs, loss, loss_pixel, loss_perceptual, loss_offset_reg

    def train_epoch(self, pbar: tqdm, loader: DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logging.debug(f"======== Starting epoch {self.current_epoch} ========")
        for data in loader:

            logging.debug(f"======== Starting global step {self.global_step} ========")
            # with self.accelerator.accumulate(self.model):

            # Move data to device
            if isinstance(data, (list, tuple)):
                data = [b.to(self.device) if torch.is_tensor(b) else b for b in data]
            elif isinstance(data, dict):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
            else:
                data = data.to(self.device)

            # forward to loss
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outs, loss, loss_pixel, loss_perceptual, loss_offset_reg = self.forward_loss_local_step(data)

            # backward
            # self.accelerator.backward(loss)
            self.backward_step(loss)

            # if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
            #     self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)

            # Gradient clipping
            if self.cfg.train.optim.get('clip_grad_norm', 0) > 0.:
                self.clip_gradients(self.cfg.train.optim.clip_grad_norm)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # track local losses
            local_step_losses.append(torch.stack([
                _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                for _loss in [loss, loss_pixel, loss_perceptual, loss_offset_reg]
            ]))

            # track global step
            # if self.accelerator.sync_gradients:
            profiler.step()
            self.scheduler.step()
            logging.debug(f"======== Scheduler step ========")
            self.global_step += 1
            # global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
            global_step_loss = gather_and_average_losses(local_step_losses)
            loss, loss_pixel, loss_perceptual, loss_offset_reg = global_step_loss.unbind()
            loss_kwargs = {
                'loss': loss.item(),
                'loss_pixel': loss_pixel.item(),
                'loss_perceptual': loss_perceptual.item(),
                'loss_offset_reg': loss_offset_reg.item(),
            }
            self.log_scalar_kwargs(
                step=self.global_step, split='train',
                **loss_kwargs
            )
            self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
            local_step_losses = []
            global_step_losses.append(global_step_loss)

            # manage display
            pbar.update(1)
            description = {
                **loss_kwargs,
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            description = '[TRAIN STEP]' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
            pbar.set_description(description)

            # periodic actions
            if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                self.save_checkpoint()
            if self.global_step % self.cfg.val.global_step_period == 0:
                self.evaluate()
                self.model.train()
            if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
                self.log_image_monitor(
                    step=self.global_step, split='train',
                    renders=outs['comp_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    gts=data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                )

            # progress control
            if self.global_step >= self.N_max_global_steps:
                # self.accelerator.set_trigger()
                self.stop_train_trigger = True
                break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_loss_pixel, epoch_loss_perceptual, epoch_loss_offset_reg = epoch_losses.unbind()
        epoch_loss_dict = {
            'loss': epoch_loss.item(),
            'loss_pixel': epoch_loss_pixel.item(),
            'loss_perceptual': epoch_loss_perceptual.item(),
            'loss_offset_reg': epoch_loss_offset_reg.item(),
        }
        self.log_scalar_kwargs(
            epoch=self.current_epoch, split='train',
            **epoch_loss_dict,
        )
        logging.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        # skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logging.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        self.evaluate()

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.is_main_process),
        ) as pbar:

            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=10, warmup=10, active=100,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                    self.cfg.logger.tracker_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                )),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) if self.cfg.logger.enable_profiler else DummyProfiler()

            with profiler:

                self.optimizer.zero_grad()
                for _ in range(self.current_epoch, self.cfg.train.epochs):

                    set_seeds(self.cfg.experiment.seed + self.current_epoch * 100, self.rank)
                    loader = self.train_data_manager.get_loader(epoch=int(self.current_epoch))

                    # loader = skipped_loader or self.train_loader
                    # skipped_loader = None

                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)

                    self._cleanup_dataloader(loader)
                    del loader
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    # if self.accelerator.check_trigger():
                    if self.stop_train_trigger:
                        break

            logging.info(f"======== Training finished at global step {self.global_step} ========")

        # final checkpoint and evaluation
        self.save_checkpoint()
        self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: Optional[int] = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        sample_data, sample_outs = None, None

        cur_epoch = epoch if epoch is not None else self.current_epoch
        loader = self.val_data_manager.get_loader(epoch=cur_epoch)

        # for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):
        for data in tqdm(loader, disable=(not self.is_main_process), total=max_val_batches):
            # Move data to device
            if isinstance(data, (list, tuple)):
                data = [b.to(self.device) if torch.is_tensor(b) else b for b in data]
            elif isinstance(data, dict):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
            else:
                data = data.to(self.device)

            if len(running_losses) >= max_val_batches:
                logging.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outs, loss, loss_pixel, loss_perceptual, loss_offset_reg = self.forward_loss_local_step(data)
            sample_data, sample_outs = data, outs

            running_losses.append(torch.stack([
                _loss if _loss is not None else torch.tensor(float('nan'), device=self.device)
                for _loss in [loss, loss_pixel, loss_perceptual, loss_offset_reg]
            ]))

        # total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
        total_losses = gather_and_average_losses(running_losses)
        total_loss, total_loss_pixel, total_loss_perceptual, total_loss_offset_reg = total_losses.unbind()
        total_loss_dict = {
            'loss': total_loss.item(),
            'loss_pixel': total_loss_pixel.item(),
            'loss_perceptual': total_loss_perceptual.item(),
            'loss_offset_reg': total_loss_offset_reg.item(),
        }

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch, split='val',
                **total_loss_dict,
            )
            logging.info(
                f'[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self.log_image_monitor(
                epoch=epoch, split='val',
                renders=sample_outs['comp_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
        else:
            self.log_scalar_kwargs(
                step=self.global_step, split='val',
                **total_loss_dict,
            )
            logging.info(
                f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self.log_image_monitor(
                step=self.global_step, split='val',
                renders=sample_outs['comp_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )

        self._cleanup_dataloader(loader)
        del loader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _cleanup_dataloader(self, loader):
        """安全清理DataLoader资源"""
        try:
            # 如果DataLoader有迭代器，尝试关闭它
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                if hasattr(loader._iterator, '_shutdown_workers'):
                    loader._iterator._shutdown_workers()
                elif hasattr(loader._iterator, 'shutdown'):
                    loader._iterator.shutdown()
                loader._iterator = None

            # 如果是accelerator包装的DataLoader
            if hasattr(loader, 'dataloader'):
                original_loader = loader.dataloader
                if hasattr(original_loader, '_iterator') and original_loader._iterator is not None:
                    if hasattr(original_loader._iterator, '_shutdown_workers'):
                        original_loader._iterator._shutdown_workers()
                    original_loader._iterator = None

        except Exception as e:
            logging.warning(f"Error during DataLoader cleanup: {e}")


def gather_and_average_losses(local_step_losses: list[torch.Tensor]) -> torch.Tensor:
    """
    在分布式环境中收集所有进程的损失张量并计算平均值。

    Args:
        local_step_losses: 一个包含当前进程中每个累积步骤的损失张量列表。

    Returns:
        一个在CPU上的张量，表示在所有进程中平均后的损失。
    """
    # 1. 将本地累积步骤的损失堆叠成一个张量
    local_losses_tensor = torch.stack(local_step_losses)

    # 2. 检查是否在分布式环境中运行
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        # 如果不是分布式环境或只有一个GPU，则直接返回本地损失
        return local_losses_tensor.cpu()

    # 3. 在分布式环境中执行收集操作
    world_size = dist.get_world_size()

    # 创建一个列表，用于存放从所有进程收集到的张量
    # 每个张量的形状和设备都应与本地张量相同
    gathered_list = [torch.zeros_like(local_losses_tensor) for _ in range(world_size)]

    # 4. 使用 all_gather 从所有进程收集张量
    # all_gather 会将每个进程的 local_losses_tensor 收集到 gathered_list 中
    dist.all_gather(gathered_list, local_losses_tensor)

    # 5. 将收集到的张量列表堆叠并计算平均值
    # torch.stack(gathered_list, dim=0) 将创建一个形状为 [world_size, num_accum_steps, ...] 的新张量
    # .mean(dim=0) 沿着进程维度（第一个维度）计算平均值
    global_step_loss = torch.stack(gathered_list, dim=0).mean(dim=0)

    # 6. 将最终结果移动到 CPU
    return global_step_loss.cpu()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    """Initialize distributed training mode from environment variables"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()

        # Set environment variables for torch.distributed
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        # Single GPU training
        rank = 0
        world_size = 1
        local_rank = 0

    return rank, world_size, local_rank


if __name__ == "__main__":
    """Main training function"""
    # Parse command line arguments for distributed training
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Config file path')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=None, help='World size for distributed training')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='12355', help='Master node port')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
    args = parser.parse_args()

    # Initialize distributed training
    rank, world_size, local_rank = init_distributed_mode()

    # Setup printing for distributed training
    setup_for_distributed(rank == 0)

    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

    # Create and run trainer
    trainer = LAMTrainer()  # or your custom trainer class

    with trainer:
        trainer.run()
