import os
import time
import math
import argparse
import shutil
import torch
import logging
import safetensors
import torch.distributed as dist

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from typing import Optional, Dict, Any, Literal
from abc import abstractmethod
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from lam.utils.seeds import set_seeds
from lam.utils.logging import configure_logger
from lam.utils.compile import configure_dynamo
from lam.runners.abstract import Runner


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./assets/config.yaml')
    args, unknown = parser.parse_known_args()
    
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    return cfg


class BasicTrainer(Runner):
    """
    A generic trainer for DDP training with multi-node support.
    
    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """
    def __init__(self, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        super().__init__()
        
        self.cfg = parse_configs()
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # DDP setup
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main_process = rank == 0
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize distributed environment
        self._init_distributed()
        
        # Set seeds
        set_seeds(self.cfg.experiment.seed, rank)
        
        # Setup logging
        self._setup_logging()
        
        # Setup directories
        self._setup_directories()
        
        # Configure compilation
        configure_dynamo(dict(self.cfg.compile))
        
        # Training attributes
        self.model: torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        
        # Training state
        self.N_max_global_steps: int = None
        self.N_global_steps_per_epoch: int = None
        self.global_step: int = 0
        self.current_epoch: int = 0
        
        # Mixed precision
        self.use_amp = self.cfg.train.get('mixed_precision', 'no') != 'no'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # TensorBoard writer
        self.writer = None
        if self.is_main_process:
            self.writer = SummaryWriter(
                log_dir=os.path.join(
                    self.cfg.logger.tracker_root,
                    self.cfg.experiment.parent,
                    self.cfg.experiment.child,
                )
            )

    def _init_distributed(self):
        """Initialize distributed training environment"""
        if self.world_size > 1:
            if not dist.is_initialized():
                # Get distributed training parameters from environment
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                dist.init_process_group(
                    backend=backend,
                    rank=self.rank,
                    world_size=self.world_size
                )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f'cuda:{self.local_rank}')

    def _setup_logging(self):
        """Setup logging configuration"""
        if self.is_main_process:
            configure_logger(
                stream_level=self.cfg.logger.stream_level,
                log_level=self.cfg.logger.log_level,
                file_path=os.path.join(
                    self.cfg.logger.log_root,
                    self.cfg.experiment.parent,
                    self.cfg.experiment.child,
                    f"{self.timestamp}.log",
                )
            )
        else:
            # Non-main processes only log warnings and errors
            logging.getLogger().setLevel(logging.WARNING)

    def _setup_directories(self):
        """Setup necessary directories"""
        self.img_log_path = os.path.join(
            self.cfg.logger.img_log_root,
            self.cfg.experiment.parent,
            self.cfg.experiment.child,
            f"{self.timestamp}",
        )
        
        if self.is_main_process:
            os.makedirs(self.img_log_path, exist_ok=True)
            os.makedirs(os.path.join(
                self.cfg.saver.checkpoint_root,
                self.cfg.experiment.parent,
                self.cfg.experiment.child
            ), exist_ok=True)

    def __enter__(self):
        """Context manager entry"""
        self.prepare_everything()
        self.log_initial_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.writer:
            self.writer.close()
        if self.world_size > 1:
            dist.destroy_process_group()

    @contextmanager
    def exec_in_order(self):
        """Execute code in process order"""
        for rank in range(self.world_size):
            if self.rank == rank:
                yield
            self.wait_for_everyone()

    def wait_for_everyone(self):
        """Synchronize all processes"""
        if self.world_size > 1:
            dist.barrier()

    def prepare_everything(self, is_dist_validation: bool = True):
        """Prepare model, optimizer, and data loaders"""
        # Build components
        # self._build_model()
        # self._build_optimizer()
        # self._build_scheduler()
        # self._build_dataloader()
        # self._build_loss_fn()
        
        # Move model to device and wrap with DDP
        self.model = self.model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=self.cfg.train.get('find_unused_parameters', False)
            )
        
        # Prepare statistics
        self.N_global_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.cfg.train.accum_steps
        )
        self.N_max_global_steps = self.N_global_steps_per_epoch * self.cfg.train.epochs
        
        if self.cfg.train.get('debug_global_steps') is not None:
            logging.warning(
                f"Overriding max global steps from {self.N_max_global_steps} "
                f"to {self.cfg.train.debug_global_steps}"
            )
            self.N_max_global_steps = self.cfg.train.debug_global_steps
        
        # Log statistics
        self._log_statistics(is_dist_validation)
        
        # Load checkpoint or model
        self.load_ckpt_or_auto_resume_(self.cfg)
        
        # Register hooks
        self.register_hooks()

    def _log_statistics(self, is_dist_validation: bool):
        """Log training statistics"""
        if not self.is_main_process:
            return
            
        logging.info("======== Statistics ========")
        logging.info(f"** World size: {self.world_size}")
        logging.info(f"** N_max_global_steps: {self.N_max_global_steps}")
        logging.info(f"** Max_img_per_gpu: {self.cfg.data.loader.max_img_per_gpu}")
        logging.info(f"** N_epochs: {self.cfg.train.epochs}")
        logging.info(f"** N_global_steps_per_epoch: {self.N_global_steps_per_epoch}")
        logging.info(f"** Distributed validation: {is_dist_validation}")
        logging.info("============================")
        
        # Log trainable parameters
        model = self.model.module if hasattr(self.model, 'module') else self.model
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("======== Trainable parameters ========")
        logging.info(f"** Total: {total_params}")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            logging.info(f"** {name}: {module_params}")
        logging.info("=====================================")

    def auto_resume_(self, cfg) -> bool:
        """Auto resume from latest checkpoint"""
        ckpt_root = os.path.join(
            cfg.saver.checkpoint_root,
            cfg.experiment.parent,
            cfg.experiment.child,
        )
        
        if not os.path.exists(ckpt_root):
            return False
            
        ckpt_dirs = [d for d in os.listdir(ckpt_root) if d.isdigit()]
        if len(ckpt_dirs) == 0:
            return False
            
        ckpt_dirs.sort(key=int)
        latest_ckpt = ckpt_dirs[-1]
        latest_ckpt_path = os.path.join(ckpt_root, latest_ckpt, "model.safetensors")
        
        if not os.path.exists(latest_ckpt_path):
            return False
            
        logging.info(f"======== Auto-resume from {latest_ckpt_path} ========")
        
        # Load model state
        model = self.model.module if hasattr(self.model, 'module') else self.model
        safetensors.torch.load_model(model, latest_ckpt_path, strict=True)
        
        # Load optimizer and scheduler state
        optimizer_path = os.path.join(ckpt_root, latest_ckpt, "optimizer.pt")
        scheduler_path = os.path.join(ckpt_root, latest_ckpt, "scheduler.pt")
        
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        if os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        
        # Load training state
        state_path = os.path.join(ckpt_root, latest_ckpt, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state.get('global_step', int(latest_ckpt))
            self.current_epoch = state.get('current_epoch', self.global_step // self.N_global_steps_per_epoch)
            if self.scaler and 'scaler' in state:
                self.scaler.load_state_dict(state['scaler'])
        else:
            self.global_step = int(latest_ckpt)
            self.current_epoch = self.global_step // self.N_global_steps_per_epoch
        
        return True

    def load_model_(self, cfg):
        """Load model from specified path"""
        logging.info(f"======== Loading model from {cfg.saver.load_model} ========")
        model = self.model.module if hasattr(self.model, 'module') else self.model
        safetensors.torch.load_model(model, cfg.saver.load_model, strict=True)
        logging.info("======== Model loaded ========")

    def load_ckpt_or_auto_resume_(self, cfg):
        """Load checkpoint or auto resume"""
        if cfg.saver.get('auto_resume', False):
            if self.auto_resume_(cfg):
                return
                
        if cfg.saver.get('load_model'):
            self.load_model_(cfg)
            return
            
        logging.debug("======== No checkpoint or model is loaded ========")

    def save_checkpoint(self):
        """Save checkpoint"""
        if not self.is_main_process:
            return
            
        ckpt_dir = os.path.join(
            self.cfg.saver.checkpoint_root,
            self.cfg.experiment.parent,
            self.cfg.experiment.child,
            f"{self.global_step:06d}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        safetensors.torch.save_model(model, os.path.join(ckpt_dir, "model.safetensors"))
        
        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }
        if self.scaler:
            state['scaler'] = self.scaler.state_dict()
            
        torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
        
        logging.info(f"======== Saved checkpoint at global step {self.global_step} ========")
        
        # Manage checkpoint cleanup
        self._cleanup_checkpoints(ckpt_dir)

    def _cleanup_checkpoints(self, current_ckpt_dir: str):
        """Clean up old checkpoints based on stratified strategy"""
        ckpt_parent_dir = os.path.dirname(current_ckpt_dir)
        ckpt_dirs = [d for d in os.listdir(ckpt_parent_dir) if d.isdigit()]
        ckpt_dirs.sort(key=int)
        
        if len(ckpt_dirs) <= 1:
            return
            
        max_ckpt = int(ckpt_dirs[-1])
        ckpt_base = int(self.cfg.saver.get('checkpoint_keep_level', 10))
        ckpt_period = self.cfg.saver.get('checkpoint_global_steps', 1000)
        
        logging.debug(f"Checkpoint base: {ckpt_base}")
        logging.debug(f"Checkpoint period: {ckpt_period}")
        
        # Ensure math.log parameter is valid
        if max_ckpt // ckpt_period <= 0:
            logging.info(f"Current max checkpoint {max_ckpt} is less than the first period {ckpt_period}. Skipping cleanup.")
            return
            
        cur_order = ckpt_base ** math.floor(math.log(max_ckpt // ckpt_period, ckpt_base))
        cur_idx = 0
        
        while cur_order > 0:
            cur_digit = max_ckpt // ckpt_period // cur_order % ckpt_base
            while cur_idx < len(ckpt_dirs) and int(ckpt_dirs[cur_idx]) // ckpt_period // cur_order % ckpt_base < cur_digit:
                if int(ckpt_dirs[cur_idx]) // ckpt_period % cur_order != 0:
                    ckpt_to_remove = os.path.join(ckpt_parent_dir, ckpt_dirs[cur_idx])
                    shutil.rmtree(ckpt_to_remove)
                    logging.info(f"Removed checkpoint {ckpt_dirs[cur_idx]}")
                cur_idx += 1
            cur_order //= ckpt_base
        
    @property
    def global_step_in_epoch(self):
        """Get current global step within epoch"""
        return self.global_step % self.N_global_steps_per_epoch
    
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1
    
    @staticmethod
    def _get_str_progress(epoch: int = None, step: int = None):
        if epoch is not None:
            log_type = 'epoch'
            log_progress = epoch
        elif step is not None:
            log_type = 'step'
            log_progress = step
        else:
            raise ValueError('Either epoch or step must be provided')
        return log_type, log_progress
    
    def log_scalar_kwargs(
            self, epoch: int = None, step: int = None,
            split: Literal['val', 'train'] = 'val', 
            **scalar_kwargs
        ):
        """Log scalar value to TensorBoard"""
        if self.writer and self.is_main_process:
            log_type, log_progress = self._get_str_progress(epoch, step)
            for key, value in scalar_kwargs.items():
                tag = f'{key}/{log_type}/{split}'
                self.writer.add_scalar(tag, value, log_progress)
    
    def log_image_monitor(
            self, epoch: int = None, step: int = None,
            split: Literal['val', 'train'] = 'val',
            renders: torch.Tensor = None, gts: torch.Tensor = None,
        ):
        """
        Log images to local dir
        Args:
            renders:    (B, V, C, H, W)
            gts:        (B, V, C, H, W)
        """
        log_type, log_progress = self._get_str_progress(epoch, step)
        img_name = f'{split}_{log_type}_{log_progress}.png'
        img_path = os.path.join(self.img_log_path, img_name)

        M = renders.shape[1]
        merged = torch.stack([renders, gts], dim=1).view(-1, *renders.shape[2:])  # (N, C, H, W)
        save_image(merged, img_path, nrow=M)
    
    def log_optimizer(self, epoch: int = None, step: int = None, attrs: list[str] = [], group_ids: list[int] = []):
        """Log optimizer state (learning rates, etc.)"""
        if self.writer and self.is_main_process:
            log_type, log_progress = self._get_str_progress(epoch, step)
            assert self.optimizer is not None, 'Optimizer is not initialized'
            if not attrs:
                logging.warning('No optimizer attributes are provided, nothing will be logged')
            if not group_ids:
                logging.warning('No optimizer group ids are provided, nothing will be logged')
            for attr in attrs:
                assert attr in ['lr', 'momentum', 'weight_decay'], f'Invalid optimizer attribute {attr}'
                for group_id in group_ids:
                    tag = f'opt/{attr}/{group_id}'
                    self.writer.add_scalar(tag, self.optimizer.param_groups[group_id][attr], log_progress)

    def log_initial_info(self):
        """Log initial training information"""
        if not self.is_main_process:
            return
        assert self.model is not None, 'Model is not initialized'
        assert self.optimizer is not None, 'Optimizer is not initialized'
        assert self.scheduler is not None, 'Scheduler is not initialized'
        
        logging.info("======== Training Configuration ========")
        logging.info(f"Config:\n{OmegaConf.to_yaml(self.cfg)}")
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        logging.info(f"Model:\n{str(model)}")
        logging.info(f"Optimizer:\n{str(self.optimizer)}")
        logging.info(f"Scheduler:\n{str(self.scheduler)}")
        logging.info("=======================================")
    
    def backward_step(self, loss: torch.Tensor):
        """Perform backward step with optional mixed precision"""
        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self):
        """Perform optimizer step with optional mixed precision"""
        if self.use_amp and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if self.use_amp and self.scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def register_hooks(self):
        """Register any necessary hooks"""
        pass
    
    @abstractmethod
    def _build_model(self):
        """Build and initialize the model"""
        pass

    @abstractmethod
    def _build_optimizer(self):
        """Build and initialize the optimizer"""
        pass

    @abstractmethod
    def _build_scheduler(self):
        """Build and initialize the learning rate scheduler"""
        pass

    @abstractmethod
    def _build_dataloader(self):
        """Build and initialize data loaders"""
        pass
    
    @abstractmethod
    def _build_loss_fn(self):
        """Build and initialize loss function"""
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def run(self):
        """Main run method"""
        self.train()
