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
import math
from numpy.random import f
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torchvision.transforms.functional import normalize
from .base_trainer import Trainer
from gstransformer.utils.profiler import DummyProfiler
from gstransformer.runners import REGISTRY_RUNNERS
from gstransformer.datasets import MixerDataset, DynamicDataManager
from torch.utils.data import DataLoader, DistributedSampler
from facexlib.parsing import init_parsing_model
from einops import rearrange
import numpy as np
logger = get_logger(__name__)
def tensor_to_pil(src_img_tensor):
    """
    Converts a tensor image to a PIL image.

    This function takes an input tensor with the shape (C, H, W) and converts it
    into a PIL Image format. It ensures that the tensor is in the correct data
    type and moves it to CPU if necessary.

    Parameters:
        src_img_tensor (torch.Tensor): Input image tensor with shape (C, H, W),
            where C is the number of channels, H is the height, and W is the width.

    Returns:
        PIL.Image: The converted image in PIL format.
    """

    img = src_img_tensor.clone().detach()
    if img.dtype == torch.bfloat16:
        img = img.to(torch.float32)
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    pil_image = Image.fromarray(img)
    return pil_image

@REGISTRY_RUNNERS.register('train.lam')
class LAMTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)

        self.train_data_manager, self.val_data_manager = self._build_dataloader(self.cfg)

        """train_loader / val_loader used for aligning interface, which will be wrapped by accelerate"""

        self.train_loader = self.train_data_manager
        self.val_loader = self.val_data_manager
        print(len(self.train_loader), "self.train_loader")
        print(len(self.val_loader), "self.val_loader")

        """real_train_loader / real_val_loader used for reading data for train / val in actual practice"""
        self.real_train_loader = self.train_data_manager 
        self.real_val_loader = self.val_data_manager 

        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self.pixel_loss_fn, self.perceptual_loss_fn, self.offset_reg_loss_fn, self.ssim_fn= self._build_loss_fn(self.cfg)
        self.mask_weight_schedule = self._parse_weight_schedule(self.cfg.train.loss.mask_weight)
        if self.cfg.train.parsing_model_dir is not None:
            parsing_model = init_parsing_model( model_name="bisenet", device=self.device, model_rootpath=self.cfg.train.parsing_model_dir )
            parsing_model = parsing_model.eval()
            self.parsing_model =parsing_model.requires_grad_(False)
        else:
            self.parsing_model = None


    def _build_model(self, cfg):
        assert cfg.experiment.type == 'lam', \
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from gstransformer.models import model_dict
        model_class = model_dict[cfg.experiment.type]
        model = model_class(**cfg.model)
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
        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        print("lrlrlrlrlrlr", cfg.train.optim.lr)
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )

        return optimizer

    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        if cfg.train.scheduler.type == 'cosine':

            from gstransformer.utils.scheduler import CosineWarmupScheduler
            print("effective_warmup_iter", effective_warmup_iters)
            print("total_global_batches", total_global_batches)
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler

    def _build_dataloader(self, cfg):
        if cfg.model.encoder_type == 'dinov2_fusion':
            PATCH_SIZE = 14
        elif cfg.model.encoder_type == 'dinov3_fusion':
            PATCH_SIZE = 16
        else:
            assert False, f"encoder_type: {self.cfg.model.encoder_type} not supported!"

        # build dataset
        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.data.subsets,
            human_model_path = cfg.data.dataset.human_model_path,
            sample_side_views=cfg.data.dataset.sample_side_views,
            source_image_res=cfg.data.dataset.source_image_res,
            render_image_res_low=cfg.data.dataset.render_image.low,
            render_image_res_high=cfg.data.dataset.render_image.high,
            render_region_size=cfg.data.dataset.render_image.region,
            repeat_num=cfg.data.dataset.repeat_num,
            aspect_standard=cfg.data.dataset.aspect_standard,
            enlarge_ratio=cfg.data.dataset.enlarge_ratio,
            multiply=PATCH_SIZE,
            use_aug = cfg.data.dataset.use_aug,
            use_ar_out = cfg.data.dataset.use_ar_out

        )
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.data.subsets,
            human_model_path = cfg.data.dataset.human_model_path,
            sample_side_views=cfg.data.dataset.sample_side_views,
            source_image_res=cfg.data.dataset.source_image_res,
            render_image_res_low=cfg.data.dataset.render_image.low,
            render_image_res_high=cfg.data.dataset.render_image.high,
            render_region_size=cfg.data.dataset.render_image.region,
            repeat_num=cfg.data.dataset.repeat_num,
            aspect_standard=cfg.data.dataset.aspect_standard,
            enlarge_ratio=cfg.data.dataset.enlarge_ratio,
            multiply=PATCH_SIZE,
            use_aug = cfg.data.dataset.use_aug,
            use_ar_out = cfg.data.dataset.use_ar_out
        )


        train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=cfg.data.loader.pin_memory,
        persistent_workers=cfg.data.loader.pin_memory,
        num_workers=cfg.data.loader.num_workers.train
        
                )
 

        val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        pin_memory=cfg.data.loader.pin_memory,
        # persistent_workers=True,
        num_workers=cfg.data.loader.num_workers.val
        )


        return train_data_loader, val_data_loader 

    def _build_loss_fn(self, cfg):
        from gstransformer.losses import PixelLoss, LPIPSLoss, OffsetReg, ssim 
        from gstransformer.losses.lpips.lpips import LPIPS
        pixel_loss_fn = PixelLoss()
        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPS(net_type='alex').to(self.device).eval()
        offset_reg_loss_fn = OffsetReg(sigma=cfg.train.loss.offset_sigma)
        return pixel_loss_fn, perceptual_loss_fn, offset_reg_loss_fn, ssim

    def register_hooks(self):
        pass

    def _update_loader_by_epoch(self, epoch: int):
        set_seed(self.cfg.experiment.seed + epoch * 100, device_specific=True)

        self.real_train_loader.batch_sampler.set_epoch(epoch)
        self.real_val_loader.batch_sampler.set_epoch(epoch)

        if hasattr(self.real_train_loader.dataset, "epoch"):
            self.real_train_loader.dataset.epoch = epoch
        if hasattr(self.real_train_loader.dataset, "set_epoch"):
            self.real_train_loader.dataset.set_epoch(epoch)
        
        if hasattr(self.real_val_loader.dataset, "epoch"):
            self.real_val_loader.dataset.epoch = epoch
        if hasattr(self.real_val_loader.dataset, "set_epoch"):
            self.real_val_loader.dataset.set_epoch(epoch)

    def forward_loss_local_step(self, data):
       

        source_c2ws = data['source_c2ws']
        source_intrs = data['source_intrs']
        render_c2ws = data['c2ws']
        render_intrs = data['intrs']

        source_image = data['source_rgbs']
        render_image = data['render_image']  # [B, N, 3, H, W]
        render_mask = data['render_mask']  # [B, N, 1, H, W]
        render_full_resolutions = data['render_full_resolutions']
        render_bg_colors = data['render_bg_colors']  # [B, N, 3]
        query_points = data["gs_xyz"]  # [B, M, 3]
        query_binding = data["gs_binding"]  # [B, M, 1]
        ar_features = data["ar_features"]
        my_lbs_weights = data["my_lbs_weights"]
        my_shapedirs_up = data["my_shapedirs_up"]
        

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
            query_points = query_points, 
            query_binding = query_binding,
            ar_features = ar_features,
            my_lbs_weights = my_lbs_weights,
            my_shapedirs_up = my_shapedirs_up

            # render_resolutions=render_full_resolutions,
            # render_region_size=self.cfg.dataset
            # .render_image.region,
        )
        
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                if not torch.all(torch.isfinite(v)):
                    
                    print(f"[!] outputs[{k}] contains NaN or Inf.")
                    raise ValueError(f"Invalid value in outputs[{k}]")

        cano_gs_lst = outputs['cano_gs_lst']
        offset_list = [_gs.offset.unsqueeze(0) for _gs in cano_gs_lst]
        offsets = torch.cat(offset_list, dim=0)
    
        # scale_list = [_gs.scale.unsqueeze(0) for _gs in cano_gs_lst]
        # scales = torch.cat(scale_list, dim=0)
        # loss calculation
        loss = 0.
        loss_pixel = None
        loss_perceptual = None
        # loss_tv = None
        loss_offset_reg = None
        loss_ssim = None

        # if self.cfg.train.loss.pixel_weight > 0.:
        #     loss_pixel = self.pixel_loss_fn(outputs['images_rgb'], render_image)
        #     loss += loss_pixel * self.cfg.train.loss.pixel_weight

        # for k, v in outputs.items():
        #     try:
        #         print(k, v.shape)
        #     except:
        #         pass
        # assert False, 'debug'

        # use input render_mask and bg_colors to calculate loss for mask region of outputs render image
        # output_rgb = outputs['comp_rgb'] * render_mask + (1 - render_mask) * render_bg_colors.unsqueeze(-1).unsqueeze(-1)

        if self.cfg.train.loss.masked_pixel_weight > 0.:
            loss_pixel = self.pixel_loss_fn(outputs['comp_rgb'], render_image)
            loss += loss_pixel * self.cfg.train.loss.masked_pixel_weight

        if self.cfg.train.loss.perceptual_weight > 0.:
            loss_perceptual = self.perceptual_loss_fn(outputs['comp_rgb'], render_image)
            loss += loss_perceptual * self.cfg.train.loss.perceptual_weight
        if self.parsing_model is not None:
            with torch.no_grad():
                gen_img_recon_parsing =  rearrange(outputs['comp_rgb'], "n m c h w -> (n m) c h w")
                parsing_out = self.parsing_model(
                    normalize(gen_img_recon_parsing, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
                parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # torch.Size([1, 1, 512, 512])
                facial_label = [2, 3, 4, 5, 6, 11, 12, 13]
                # bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
                facial = sum(parsing_out == i for i in facial_label).bool()
                facial_mask = facial.to(torch.float)
                facial_mask = rearrange(facial_mask, "(n m) c h w -> n m c h w", n=outputs['comp_rgb'].shape[0])

                bg = sum(parsing_out == i for i in facial_label).bool()
                float_mask = bg.to(torch.float)

                white_image = torch.ones_like(gen_img_recon_parsing)  # torch.Size([1, 3, 512, 512])
                # only keep the face features
                return_face_features_image_2 = torch.where(~bg, white_image, input)  # torch.Size([1, 3, 512, 512])
                return_face_features_image_2 = gen_img_recon_parsing*float_mask
                return_face_features_image_2_pil = tensor_to_pil(return_face_features_image_2[0]*255.0)
                return_face_features_image_2_pil.save('parsing_test.png')
            gen_img_recon_facial = outputs['comp_rgb'] * facial_mask
            real_img_recon_facial = render_image * facial_mask
            loss_pixel_parsing = self.pixel_loss_fn(gen_img_recon_facial, real_img_recon_facial)
            loss += loss_pixel_parsing * self.cfg.train.loss.masked_pixel_weight * 20




        # if self.cfg.train.loss.tv_weight > 0.:
        #     loss_tv = self.tv_loss_fn(outputs['planes'])
        #     loss += loss_tv * self.cfg.train.loss.tv_weight
        
        if self.cfg.train.loss.offset_reg_weight > 0.:
            loss_offset_reg = self.offset_reg_loss_fn(offsets)
            loss += loss_offset_reg * self.cfg.train.loss.offset_reg_weight
        # if self.cfg.train.loss.scales_reg_weight > 0.:
        #     loss_sacle_reg = self.scale_reg(scales, outputs['visibility_filter'] )
        #     loss += loss_sacle_reg * self.cfg.train.loss.scales_reg_weight
        
        if self.cfg.train.loss.masked_pixel_weight > 0.:
            loss_pixel = self.pixel_loss_fn(outputs['comp_rgb'], render_image)
            loss += loss_pixel * self.cfg.train.loss.masked_pixel_weight
        if self.cfg.train.loss.ssim_weight > 0.:
            loss_ssim = 1-self.ssim_fn(outputs['comp_rgb'], render_image)
            loss += loss_ssim * self.cfg.train.loss.ssim_weight 
        return outputs, loss, loss_pixel, loss_perceptual, loss_offset_reg

    def train_epoch(self, pbar: tqdm, loader: DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        for data in loader:

            # data = torch.load("/robby/share/Avatar/hongyul/code/avatarpointillist/avatarpointillist/debug_data.pth")
            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):
                # Move data to device
                if isinstance(data, (list, tuple)):
                    data = [b.to(self.device) if torch.is_tensor(b) else b for b in data]
                elif isinstance(data, dict):
                    data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
                else:
                    data = data.to(self.device)



                # forward to loss
                outs, loss, loss_pixel, loss_perceptual, loss_offset_reg = self.forward_loss_local_step(data)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                

                # track local losses
                local_step_losses.append(torch.stack([
                    _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                    for _loss in [loss, loss_pixel, loss_perceptual, loss_offset_reg]
                ]))

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
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
                    self.accelerator.set_trigger()
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
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        # skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        self.evaluate()

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.accelerator.is_main_process),
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
                    # self._update_loader_by_epoch(self.current_epoch + self.accelerator.process_index)
                    self.train_epoch(pbar=pbar, loader=self.real_train_loader, profiler=profiler)
                    if self.accelerator.check_trigger():
                        break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

            # final checkpoint and evaluation
            self.save_checkpoint()
            self.evaluate()

    @torch.no_grad()
    # @torch.compiler.disable
    def evaluate(self, epoch: Optional[int] = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        sample_data, sample_outs = None, None

        cur_epoch = epoch if epoch is not None else self.current_epoch
        # self._update_loader_by_epoch(cur_epoch + self.accelerator.process_index)

        # for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):
        for data in tqdm(self.real_val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):
            # Move data to device
            if isinstance(data, (list, tuple)):
                data = [b.to(self.device) if torch.is_tensor(b) else b for b in data]
            elif isinstance(data, dict):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
            else:
                data = data.to(self.device)

            if len(running_losses) >= max_val_batches:
                logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            outs, loss, loss_pixel, loss_perceptual, loss_offset_reg = self.forward_loss_local_step(data)
            sample_data, sample_outs = data, outs

            running_losses.append(torch.stack([
                _loss if _loss is not None else torch.tensor(float('nan'), device=self.device)
                for _loss in [loss, loss_pixel, loss_perceptual, loss_offset_reg]
            ]))

        total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
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
            logger.info(
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
            logger.info(
                f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self.log_image_monitor(
                step=self.global_step, split='val',
                renders=sample_outs['comp_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )

    def _parse_weight_schedule(self, weight_config):
        """
        解析权重调度配置
        Args:
            weight_config: str, 格式为 "start_step:start_weight:end_weight:end_step"
        Returns:
            dict: 包含调度参数的字典
        """
        if isinstance(weight_config, (int, float)):
            # 如果是单个数值，返回固定权重
            return {
                'type': 'fixed',
                'weight': float(weight_config)
            }
        
        parts = str(weight_config).split(':')
        if len(parts) != 4:
            raise ValueError(f"Invalid weight schedule format: {weight_config}")
        
        start_step, start_weight, end_weight, end_step = parts
        return {
            'type': 'linear',
            'start_step': int(start_step),
            'start_weight': float(start_weight),
            'end_weight': float(end_weight),
            'end_step': int(end_step)
        }

    def _get_scheduled_weight(self, weight_schedule, current_step):
        """
        根据当前训练步数获取权重
        Args:
            weight_schedule: dict, 权重调度配置
            current_step: int, 当前训练步数
        Returns:
            float: 当前步数对应的权重
        """
        if weight_schedule['type'] == 'fixed':
            return weight_schedule['weight']
        
        elif weight_schedule['type'] == 'linear':
            start_step = weight_schedule['start_step']
            start_weight = weight_schedule['start_weight']
            end_weight = weight_schedule['end_weight']
            end_step = weight_schedule['end_step']
            
            if current_step <= start_step:
                return start_weight
            elif current_step >= end_step:
                return end_weight
            else:
                # 线性插值
                progress = (current_step - start_step) / (end_step - start_step)
                return start_weight + (end_weight - start_weight) * progress
        
        else:
            raise ValueError(f"Unknown weight schedule type: {weight_schedule['type']}")
