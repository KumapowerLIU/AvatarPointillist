import os
import time
import math
import argparse
import shutil
import torch
import logging
import safetensors
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any, Union
from omegaconf import OmegaConf
from abc import abstractmethod
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter

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


class DDPTrainer(Runner):
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
        set_seeds(self.cfg.experiment.seed + rank)
        
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
        self.loss_fn = None
        
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
                    self.timestamp
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

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes for metrics calculation"""
        if self.world_size == 1:
            return tensor
        
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)
    
    def reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Reduce tensor across all processes"""
        if self.world_size == 1:
            return tensor
        
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op)
        if op == dist.ReduceOp.SUM:
            tensor = tensor / self.world_size
        return tensor
    
    def prepare_everything(self, is_dist_validation: bool = True):
        """Prepare model, optimizer, and data loaders"""
        # Build components
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._build_dataloader()
        self._build_loss_fn()
        
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
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value to TensorBoard"""
        if self.writer and self.is_main_process:
            step = step if step is not None else self.global_step
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar values"""
        if self.writer and self.is_main_process:
            step = step if step is not None else self.global_step
            for tag, value in tag_scalar_dict.items():
                self.writer.add_scalar(tag, value, step)

    def log_images(self, tag: str, images: torch.Tensor, step: Optional[int] = None):
        """Log images to TensorBoard"""
        if self.writer and self.is_main_process:
            step = step if step is not None else self.global_step
            self.writer.add_images(tag, images, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram to TensorBoard"""
        if self.writer and self.is_main_process:
            step = step if step is not None else self.global_step
            self.writer.add_histogram(tag, values, step)

    def log_optimizer_state(self, step: Optional[int] = None):
        """Log optimizer state (learning rates, etc.)"""
        if not self.is_main_process:
            return
            
        step = step if step is not None else self.global_step
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log_scalar(f'optimizer/lr_group_{i}', param_group['lr'], step)
            if 'momentum' in param_group:
                self.log_scalar(f'optimizer/momentum_group_{i}', param_group['momentum'], step)
            if 'weight_decay' in param_group:
                self.log_scalar(f'optimizer/weight_decay_group_{i}', param_group['weight_decay'], step)

    def log_model_gradients(self, step: Optional[int] = None):
        """Log model gradient norms"""
        if not self.is_main_process:
            return
            
        step = step if step is not None else self.global_step
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                self.log_scalar(f'gradients/{name}', param_norm, step)
        
        total_norm = total_norm ** (1. / 2)
        self.log_scalar('gradients/total_norm', total_norm, step)

    def log_initial_info(self):
        """Log initial training information"""
        if not self.is_main_process:
            return
            
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

    def training_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Single training step - to be implemented by subclass"""
        raise NotImplementedError("Subclass must implement training_step")
    
    def validation_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Single validation step - to be implemented by subclass"""
        raise NotImplementedError("Subclass must implement validation_step")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {}
        accumulated_loss = 0.0
        num_batches = 0
        
        # Set sampler epoch for proper shuffling
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                step_output = self.training_step(batch)
                loss = step_output['loss'] / self.cfg.train.accum_steps
            
            # Backward pass
            self.backward_step(loss)
            accumulated_loss += loss.item()
            
            # Optimizer step
            if (batch_idx + 1) % self.cfg.train.accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Gradient clipping
                if self.cfg.train.get('max_grad_norm', 0) > 0:
                    self.clip_gradients(self.cfg.train.max_grad_norm)
                
                # Optimizer step
                self.optimizer_step()
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler and self.cfg.train.get('scheduler_step_per_batch', False):
                    self.scheduler.step()
                
                # Increment global step
                self.global_step += 1
                
                # Logging
                if self.global_step % self.cfg.logger.get('log_every_n_steps', 100) == 0:
                    # Reduce metrics across processes
                    reduced_loss = self.reduce_tensor(torch.tensor(accumulated_loss, device=self.device))
                    
                    self.log_scalar('train/loss', reduced_loss.item(), self.global_step)
                    self.log_optimizer_state(self.global_step)
                    
                    # Log additional metrics from training step
                    for key, value in step_output.items():
                        if key != 'loss' and torch.is_tensor(value):
                            reduced_value = self.reduce_tensor(value)
                            self.log_scalar(f'train/{key}', reduced_value.item(), self.global_step)
                    
                    if self.cfg.logger.get('log_gradients', False):
                        self.log_model_gradients(self.global_step)
                
                # Save checkpoint
                if (self.cfg.saver.get('checkpoint_global_steps', 1000) > 0 and 
                    self.global_step % self.cfg.saver.checkpoint_global_steps == 0):
                    self.save_checkpoint()
                
                # Reset accumulated loss
                accumulated_loss = 0.0
                
                # Early stopping for debug
                if (self.cfg.train.get('debug_global_steps') and 
                    self.global_step >= self.cfg.train.debug_global_steps):
                    break
            
            num_batches += 1
        
        # Scheduler step per epoch
        if self.scheduler and not self.cfg.train.get('scheduler_step_per_batch', False):
            self.scheduler.step()
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_metrics = {}
        all_outputs = []
        
        # Set sampler epoch
        if hasattr(self.val_loader.sampler, 'set_epoch'):
            self.val_loader.sampler.set_epoch(self.current_epoch)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    step_output = self.validation_step(batch)
                
                all_outputs.append(step_output)
        
        # Aggregate metrics
        if all_outputs:
            for key in all_outputs[0].keys():
                if torch.is_tensor(all_outputs[0][key]):
                    values = torch.stack([output[key] for output in all_outputs])
                    # Gather from all processes
                    all_values = self.gather_for_metrics(values)
                    val_metrics[key] = all_values.mean().item()
        
        # Log validation metrics
        for key, value in val_metrics.items():
            self.log_scalar(f'val/{key}', value, self.global_step)
        
        return val_metrics
    
    def train(self):
        """Main training loop"""
        logging.info("======== Starting Training ========")
        
        for epoch in range(self.current_epoch, self.cfg.train.epochs):
            self.current_epoch = epoch
            
            logging.info(f"Epoch {epoch + 1}/{self.cfg.train.epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            if (self.cfg.train.get('val_every_n_epochs', 1) > 0 and 
                (epoch + 1) % self.cfg.train.val_every_n_epochs == 0):
                val_metrics = self.validate_epoch()
                logging.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint at end of epoch
            if (self.cfg.saver.get('checkpoint_every_n_epochs', 0) > 0 and 
                (epoch + 1) % self.cfg.saver.checkpoint_every_n_epochs == 0):
                self.save_checkpoint()
            
            # Early stopping check
            if self.global_step >= self.N_max_global_steps:
                logging.info("Reached maximum global steps, stopping training")
                break
        
        logging.info("======== Training Completed ========")

    def evaluate(self):
        """Run evaluation only"""
        logging.info("======== Starting Evaluation ========")
        val_metrics = self.validate_epoch()
        logging.info(f"Evaluation results: {val_metrics}")
        return val_metrics
    
    def run(self):
        """Main run method"""
        self.train()

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


def setup_distributed_sampler(dataset, world_size: int, rank: int, shuffle: bool = True):
    """Setup distributed sampler for dataset"""
    if world_size > 1:
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
    return None


def launch_distributed_training(
        train_fn,
        cfg_path: str = None,
        world_size: int = None,
        master_addr: str = "localhost",
        master_port: str = "12355"
    ):
    """
    Launch distributed training across multiple GPUs/nodes
    
    Args:
        train_fn: Training function that takes (rank, world_size, local_rank) as arguments
        cfg_path: Path to configuration file
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
    """
    # Auto-detect world size if not provided
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if world_size == 1:
        # Single GPU/CPU training
        train_fn(rank=0, world_size=1, local_rank=0)
    else:
        # Multi-GPU training
        mp.spawn(
            train_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )


def distributed_train_worker(rank: int, world_size: int, local_rank: int = None):
    """
    Distributed training worker function
    
    Args:
        rank: Global rank of the process
        world_size: Total number of processes
        local_rank: Local rank within the node (defaults to rank for single-node)
    """
    if local_rank is None:
        local_rank = rank
    
    # Set environment variables
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    # Create trainer instance
    trainer = DDPTrainer(rank=rank, world_size=world_size, local_rank=local_rank)
    
    # Run training
    with trainer:
        trainer.run()


# Example concrete implementation
class ExampleTrainer(DDPTrainer):
    """Example implementation of DDPTrainer"""
    
    def register_hooks(self):
        """Register training hooks"""
        pass
    
    def _build_model(self):
        """Build model - example implementation"""
        # This should be implemented based on your specific model
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        self.model = SimpleModel()
    
    def _build_optimizer(self):
        """Build optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.get('weight_decay', 0.01)
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.train.epochs,
            eta_min=self.cfg.train.get('min_lr', 1e-6)
        )
    
    def _build_dataloader(self):
        """Build data loaders"""
        from torch.utils.data import TensorDataset
        
        # Example dummy data
        train_data = torch.randn(1000, 10)
        train_targets = torch.randn(1000, 1)
        val_data = torch.randn(200, 10)
        val_targets = torch.randn(200, 1)
        
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        
        # Setup distributed samplers
        train_sampler = setup_distributed_sampler(
            train_dataset, self.world_size, self.rank, shuffle=True
        )
        val_sampler = setup_distributed_sampler(
            val_dataset, self.world_size, self.rank, shuffle=False
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.data.loader.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.data.loader.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.data.loader.get('val_batch_size', self.cfg.data.loader.batch_size),
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.cfg.data.loader.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
    
    def _build_loss_fn(self):
        """Build loss function"""
        self.loss_fn = torch.nn.MSELoss()
    
    def training_step(self, batch) -> Dict[str, torch.Tensor]:
        """Single training step"""
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        # Calculate additional metrics
        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(outputs, targets)
            mae = torch.nn.functional.l1_loss(outputs, targets)
        
        return {
            'loss': loss,
            'mse': mse,
            'mae': mae
        }
    
    def validation_step(self, batch) -> Dict[str, torch.Tensor]:
        """Single validation step"""
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        # Calculate metrics
        mse = torch.nn.functional.mse_loss(outputs, targets)
        mae = torch.nn.functional.l1_loss(outputs, targets)
        
        return {
            'loss': loss,
            'mse': mse,
            'mae': mae
        }
    

# Utility functions for multi-node training
def get_dist_info():
    """Get distributed training information from environment variables"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


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


# Main training script example
def main():
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
    trainer = ExampleTrainer()  # or your custom trainer class
    
    with trainer:
        trainer.run()


# Alternative launch methods
def launch_with_torchrun():
    """
    Example of how to use with torchrun:
    
    # Single node, multiple GPUs:
    torchrun --nproc_per_node=4 train_script.py --config config.yaml
    
    # Multiple nodes:
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_script.py
    """
    main()


def launch_with_slurm():
    """
    Example SLURM script:
    
    #!/bin/bash
    #SBATCH --job-name=ddp_training
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gres=gpu:4
    #SBATCH --cpus-per-task=8
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=12355
    
    srun python train_script.py --config config.yaml
    """
    main()


if __name__ == "__main__":
    # Different launch methods based on environment
    if 'SLURM_PROCID' in os.environ:
        # Running with SLURM
        launch_with_slurm()
    elif 'LOCAL_RANK' in os.environ:
        # Running with torchrun
        launch_with_torchrun()
    else:
        # Manual launch or single GPU
        main()
