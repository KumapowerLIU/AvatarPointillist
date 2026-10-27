import os
import argparse
import logging
from omegaconf import OmegaConf
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import datetime
from accelerate.utils import DistributedDataParallelKwargs

from avatarpointillist.argeometry.dataset.datasetgpt import DataHeadARGPT

from avatarpointillist.argeometry.engine import do_train
from transformers import Dinov2Model, AutoImageProcessor
from point_encoder.miche.encode import load_model
from avatarpointillist.argeometry.Transformer.autoregressive_wrapper import AutoregressiveWrapper
from avatarpointillist.argeometry.Transformer.decoder import Decoder
from avatarpointillist.argeometry.Transformer.transformerwarper import TransformerWrapper
# torch.backends.cudnn.enabled = False
def get_args_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser('Human AR Model Training', add_help=False)
    parser.add_argument('--config_dir', type=str, required=True, help='Path to config file')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained model weights')
    parser.add_argument('--mask_prob', default=0, type=float, help='Mask probability for GPT model')
    parser.add_argument('--checkpoint_dir', type=str, default="/robby/share/Avatar/hongyul/code/avatarpointillist/checkpoint", help='Directory to save checkpoints')
    return parser


def collate_fn(data):
    """Batch processing function"""
    dino_imgs = torch.stack([example["dino_img"] for example in data], dim=0)
    sequences = torch.stack([torch.from_numpy(example["sequence"]) for example in data], dim=0)
    pc_normals = torch.stack([torch.from_numpy(example["pc_normal"]) for example in data], dim=0)
    return {
        "dino_imgs": dino_imgs,
        "sequences": sequences,
        "pc_normals": pc_normals
    }


def setup_logging(args):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = get_logger(__file__)
    return logger


def setup_accelerator(args):
    """Setup Accelerator"""
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.precision,
        log_with="tensorboard",
        project_dir=args.checkpoint_dir,
        kwargs_handlers=[kwargs]
    )
    
    return accelerator


def setup_checkpoint_dir(args):
    """Setup checkpoint directory"""
    cur_time = datetime.datetime.now().strftime("%m_%d_%H-%M-%S")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, cur_time)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return cur_time


def setup_tensorboard(accelerator, config, cur_time):
    """Setup TensorBoard tracker"""
    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(config.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{cur_time}")


def create_dataset(args, device ):
    """Create dataset"""
    print("Creating dataset...")
    train_dataset = DataHeadARGPT(args, device=device)
    print(f"Dataset size: {len(train_dataset)}")
    return train_dataset


def create_dataloader(train_dataset, args):
    """Create data loader"""
    print("Creating dataloader...")
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batchsize_per_gpu,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True,
    )
    return dataloader 


def create_models(args, mask_prob ):
    """Create models"""
    print("Creating GPT model...")
    
    # Create AR training model
    atten_layers = Decoder(
        dim = 1024,
        depth = 24,
        heads = 16,
        cross_attend = True,
        rotary_pos_emb = True,
        attn_flash = True,
        use_rmsnorm = True
    )
    base_model= TransformerWrapper(
            num_tokens = args.num_faces + args.n_discrete_size + 3 ,
            max_seq_len = args.window_size,
            attn_layers = atten_layers
        )
    model = AutoregressiveWrapper(
            base_model, pad_value=args.num_faces + args.n_discrete_size + 2, mask_prob=mask_prob)
 
    
    # Create DINO model
    dinov2 = Dinov2Model.from_pretrained(args.dino_path)
    dinov2.requires_grad_(False)

    point_encoder = load_model(args.point_encoder_ckpt_path)
    point_encoder.requires_grad_(False)
    # Create DINO image processor
    dino_img_processor = AutoImageProcessor.from_pretrained(args.dino_path)
    
    print(f"AR Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"DINO Model parameters: {sum(p.numel() for p in dinov2.parameters()):,}")
    
    return model, dinov2, dino_img_processor, point_encoder


def main():
    """Main function"""
    # Parse command line arguments
    parser = get_args_parser()
    args = parser.parse_args()

    mask_prob = args.mask_prob

    # Load config file
    print(f"Loading config from: {args.config_dir}")
    config = OmegaConf.load(args.config_dir)
    # Merge command line arguments and config file
    config.pretrained_weights = args.pretrained_weights
    config.checkpoint_dir = args.checkpoint_dir
    
 
    # Convert to Namespace object
    args = argparse.Namespace(**config)
    
    # Setup logging
    logger = setup_logging(args)
    
    # Setup checkpoint directory
    cur_time = setup_checkpoint_dir(args)
    
    # Setup Accelerator
    accelerator = setup_accelerator(args)
    
    # Setup TensorBoard
    setup_tensorboard(accelerator, config, cur_time)
    
    # Set random seed
    set_seed(args.seed, device_specific=True)
    
    # Create dataset
    train_dataset = create_dataset(args, accelerator.device )
    
    # Create data loader
    dataloader = create_dataloader(train_dataset, args)
    
    # Create models
    model, dinov2, dino_img_processor, point_encoder = create_models(args,   mask_prob)
    
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("TRAINING CONFIGURATION")
        print("="*50)
        print(f"Batch size per GPU: {args.batchsize_per_gpu}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Precision: {args.precision}")
        print(f"Checkpoint directory: {args.checkpoint_dir}")
        print(f"Dataset size: {len(train_dataset)}")
        if hasattr(args, 'spatial_bounds'):
            print(f"Spatial bounds: {args.spatial_bounds}")
        if hasattr(args, 'n_discrete_size'):
            print(f"Coordinate bins: {args.n_discrete_size}")
        if hasattr(args, 'binding_offset'):
            print(f"Binding offset: {args.binding_offset}")
        print("="*50 + "\n")
    
    # Prepare data loader dictionary
    dataloaders = {'train': dataloader}
    # Start training
    try:
        do_train(
            args, 
            model, 
            dinov2, 
            point_encoder,
            dataloaders, 
            logger, 
            accelerator
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e


if __name__ == "__main__":
    main()
