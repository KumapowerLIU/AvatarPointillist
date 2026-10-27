import os
import torch
import sys
current_file = os.path.abspath(__file__)
project_root = os.path.join(current_file, 'avatarpointillist')
sys.path.append(project_root)


torch.autograd.set_detect_anomaly(True)
import numpy as np
from PIL import Image
import argparse
from omegaconf import OmegaConf
from transformers import Dinov2Model, AutoImageProcessor
from point_encoder.miche.encode import load_model
from torch.utils.data import DataLoader
from tqdm import tqdm

# from avatarpointillist.argeometry.TruncatedTrain import HumanAR_training
import random
import numpy as np
import torch 
from avatarpointillist.argeometry.Transformer.autoregressive_wrapper import AutoregressiveWrapper
from avatarpointillist.argeometry.Transformer.decoder import Decoder
from avatarpointillist.argeometry.Transformer.transformerwarper import TransformerWrapper
from avatarpointillist.gstransformer.models.rendering.flame_model.flame_GSA import FlameHead
from avatarpointillist.argeometry.dataset.datasetgpt_test import DataHeadARGPTTest



def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
        deterministic (bool): Whether to enable deterministic mode (affects CUDNN).

    Usage:
        set_seed(2025)
    """
    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set for multi-GPU as well

 

 

    print(f"[INFO] All seeds set to {seed}, deterministic={deterministic}")

def load_test_model(args, device):
    """Load test models"""
    print("Loading models...")

    # Create Hourglass Transformer model
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
                max_seq_len = 9600,
                attn_layers = atten_layers
            )
    # 8000 for original, 9600 for bug fix
    model = AutoregressiveWrapper(
                base_model, pad_value=args.num_faces + args.n_discrete_size + 2)
    
    # # Load pretrained weights
    if args.pretrained_weights:
        print(f"Loading pretrained weights from: {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        
        # Load model weights (use strict=False to ignore KV Cache parameters)
        model.load_state_dict(checkpoint["model"])
        print("Model weights loaded successfully!")
    
    model.to(device)
    model.eval()
    model=torch.compile(model)
    # Load DINO model
    dino_model = Dinov2Model.from_pretrained(args.dino_path)
    dino_model.requires_grad_(False)
    dino_model.to(device=device, dtype=torch.float32)
    
    dino_model.eval()
    
    # Load point cloud encoder
    point_encoder = load_model(args.point_encoder_ckpt_path)
    point_encoder.requires_grad_(False)
    point_encoder.to(device)
    point_encoder.eval()
    
    # Create DINO image processor
    dino_img_processor = AutoImageProcessor.from_pretrained(args.dino_path)
    
    return model, dino_model, point_encoder, dino_img_processor


def process_condition_data(batch_data, dino_model, point_encoder, device, weight_dtype=torch.float32):
    """Process condition data"""
     # Process DINO features
    face_dino_imgs = batch_data["dino_imgs"]

    face_features = dino_model(
        face_dino_imgs.to(device)
    ).last_hidden_state
     # Process point cloud features
    pc_feature = point_encoder.encode_latents(
        batch_data["pc_normals"].to(device)
    )
    shape_latents = point_encoder.to_shape_latents(pc_feature[:, 1:])
    
    return face_features, pc_feature, shape_latents
def get_type_mask(tokens_to_generate, device, xyz_size=1024, face_nums=10144, group_size=4):
    end_idx = xyz_size + face_nums + 1
    vocab_size = xyz_size + face_nums + 3  # +3 for start, end, pad tokens
    type2mask = torch.full((tokens_to_generate, vocab_size), float("-inf"), device=device)

    for i in range(tokens_to_generate):
        # Key: i % group_size maps i to cyclic sequence [0, 1, 2, 3]
        position_in_group = i % group_size
        
        # First, second, third tokens can only be xyz type
        if position_in_group in [0, 1, 2]:
            type2mask[i, :xyz_size] = 0.0
        # Fourth token can only be face type
        elif position_in_group == 3:
            type2mask[i, xyz_size:xyz_size + face_nums] = 0.0
            
        # Additionally allow end token, which can appear at any position
        type2mask[i, end_idx] = 0.0

    return type2mask
def from_index_to_point(generated_ids, coord_bins=1024, bound_lower=None, bound_upper=None, output_path=None):
    """
    Convert generated token sequence to point cloud coordinates
    
    Args:
        generated_ids: Generated token sequence (batch_size, seq_len)
        coord_bins: Coordinate quantization level, default 512
        bound_lower: Global minimum boundary for denormalization
        bound_upper: Global maximum boundary for denormalization
        output_path: Path to save PLY file
    
    Returns:
        xyz_coords: Point cloud coordinates (N, 3)
        binding_values: Binding values (N,)
    """
    from avatarpointillist.argeometry.dataset.data_utils import save_ply
    
    generated_ids = generated_ids.cpu().numpy()
    batch_size, seq_len = generated_ids.shape
    
    # Subtract offset to restore original token range
    generated_ids = generated_ids 
    
    # Group by 4, discard incomplete groups
    num_groups = seq_len // 4
    if num_groups == 0:
        print("Warning: No complete groups found in generated sequence")
        return None, None
    
    # Reshape to (batch_size, num_groups, 4)
    grouped_ids = generated_ids[:, :num_groups * 4].reshape(batch_size, num_groups, 4)
    
    # Process each batch
    all_xyz_coords = []
    all_binding_values = []
    
    for batch_idx in range(batch_size):
        batch_groups = grouped_ids[batch_idx]  # (num_groups, 4)
        
        # Skip first group (start token group)
        if num_groups > 1:
            point_groups = batch_groups[1:]  # Skip first group
        else:
            point_groups = batch_groups  # If only one group, also process
        
        xyz_coords = []
        binding_values = []
        
        for group in point_groups:
            # First 3 in each group are xyz coordinates, 4th is binding
            x_idx, y_idx, z_idx, binding_idx = group
            
            # Check if EOS token (1-2 = -1)
            if x_idx == -1 or y_idx == -1 or z_idx == -1 or binding_idx == -1:
                break  # Stop processing when encountering EOS token
            
            # Check if coordinates are in valid range [0, coord_bins-1]
            if (0 <= x_idx < coord_bins and 
                0 <= y_idx < coord_bins and 
                0 <= z_idx < coord_bins):
                
                # Convert quantized coordinates to normalized coordinates in [0,1] range
                x_norm = x_idx / (coord_bins - 1)
                y_norm = y_idx / (coord_bins - 1)
                z_norm = z_idx / (coord_bins - 1)
                
                xyz_coords.append([x_norm, y_norm, z_norm])
                binding_values.append(binding_idx-coord_bins)
        
        if len(xyz_coords) > 0:
            xyz_coords = np.array(xyz_coords)
            binding_values = np.array(binding_values)
            
            # Denormalize to original coordinate range
            if bound_lower is not None and bound_upper is not None:
                xyz_coords = xyz_coords * (bound_upper - bound_lower) + bound_lower
            
            all_xyz_coords.append(xyz_coords)
            all_binding_values.append(binding_values)
            
            print(f"Batch {batch_idx}: Generated {len(xyz_coords)} points")
            
            # Save PLY file
            if output_path is not None:
                ply_path = f"{output_path}_batch_{batch_idx}.ply"
                save_ply(ply_path, xyz_coords)
                print(f"Point cloud saved to: {ply_path}")
        else:
            print(f"Batch {batch_idx}: No valid points generated")
            all_xyz_coords.append(np.array([]).reshape(0, 3))
            all_binding_values.append(np.array([]))
    
    return all_xyz_coords, all_binding_values
    
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



def main():
    set_seed(seed=0)
    """Main function"""
    parser = argparse.ArgumentParser('Human AR Model Testing', add_help=False)
    parser.add_argument('--config_dir', type=str, default='/nas5/liuhongyu/AR_Results/cvpr2026_code_review/avatarpointillist/configs/geometry_config/ar_testing.yaml', help='Path to config file')
    parser.add_argument('--pretrained_weights', type=str, default='/nas5/liuhongyu/AR_Results/cvpr2026_code_review/pretrain_model/checkpoint_ar.pth', help='Path to pretrained model weights')
    # parser.add_argument('--test_data_path', type=str, required=True, help='Test data path')
    parser.add_argument('--output_dir', type=str, default='../test_outputs_ar', help='Output directory')
    parser.add_argument('--max_length', type=int, default=120000, help='Maximum generation length')
    parser.add_argument('--window_size', type=int, default=10000, help='window length')
    parser.add_argument('--file_path_test', type=str, default=None, help='Test data path')

    parser.add_argument('--topk', type=int, default=1, help='topk sampling')
    parser.add_argument('--topp', type=float, default=0.95, help='topp')
    parser.add_argument('--temp', type=float, default=0.0, help='temperature')
    # parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    
    args = parser.parse_args()
    
    # Load config file
    print(f"Loading config from: {args.config_dir}")
    config = OmegaConf.load(args.config_dir)
    
    # Merge command line arguments and config file
    config.pretrained_weights = args.pretrained_weights
    # config.test_data_path = args.test_data_path
    config.output_dir = args.output_dir
    config.window_size = args.window_size
    config.max_length = args.max_length
    config.topk = args.topk
    config.topp = args.topp
    config.temp = args.temp
    if args.file_path_test is not None:
        config.file_path_train = args.file_path_test
    # config.batch_size = args.batch_size
    
    # Convert to Namespace object
    args = argparse.Namespace(**config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    human_model_path =args.flame_base_path
    flame_model_new  = FlameHead(
        300,
        100,
        flame_model_path=f'{human_model_path}/flame2023.pkl',
        flame_lmk_embedding_path=f"{human_model_path}/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path=f"{human_model_path}/head_template_mesh.obj",
        flame_mask_path=f"{human_model_path}/FLAME_masks.pkl"

    )
    flame_model_new = flame_model_new.cuda()
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    model, dino_model, point_encoder, dino_img_processor = load_test_model(args, device)
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = DataHeadARGPTTest(args, device=device, test=True)
    
    # Create data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of test batches: {len(test_dataloader)}")
    
    # Start testing
    print("Starting test...")
    model.eval()
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float32):

        for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Processing batches", unit="batch")):
            tqdm.write(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")
            output_path = os.path.join(args.output_dir, f"topk_{str(args.topk)}_topp_{str(args.topp)}_temp_{str(args.temp)}.npy")
            output_path_embe = os.path.join(args.output_dir, f"topk_{str(args.topk)}_topp_{str(args.topp)}_temp_{str(args.temp)}_embed.npy")

            # Process condition data
            face_features, pc_feature, shape_latents = process_condition_data(
                batch_data, dino_model, point_encoder, device
            )


            # Generate sequence
            prompts = batch_data['sequences'][:, :4000].to(device)  # Use first 4 tokens as prompt
            type_mask = get_type_mask(args.max_length,  device)
            generated_ids, out_embedding = model.generate(prompts, args.max_length, temperature=args.temp,  type2mask=type_mask,filter_kwargs={"k": args.topk},
                face_features=face_features, pc_feature=pc_feature, shape_latents= shape_latents 
            )


            # generated_ids = batch_data['sequences'].to(device)           
            np.save(output_path, generated_ids.cpu().numpy())
            np.save(output_path_embe, out_embedding.cpu().numpy())
            tqdm.write(f"Generated sequence saved to: {output_path}")
            tqdm.write(f"Generated sequence shape: {generated_ids.shape}")
            tqdm.write(f"Generated emb shape: {out_embedding.shape}")
            tqdm.write(f"Generated sequence length: {generated_ids.size(1)}")
            tqdm.write("Converting tokens to point cloud...")
            xyz_coords_list, binding_values_list = from_index_to_point(
                generated_ids,
                coord_bins=args.n_discrete_size,
                bound_lower=np.array(args.global_min, dtype=np.float32),
                bound_upper=np.array(args.global_max, dtype=np.float32),
                output_path=os.path.join(args.output_dir, f"topk_{str(args.topk)}_topp_{str(args.topp)}_temp_{str(args.temp)}")
            )
            

            for i, (xyz_coords, binding_values) in enumerate(zip(xyz_coords_list, binding_values_list)):
                if len(xyz_coords) > 0:
                    point_cloud_data = {
                        'xyz': xyz_coords,
                        'binding': binding_values
                    } 
                    with torch.no_grad():
                        gs_xyz = torch.from_numpy(point_cloud_data['xyz']).float().to(device)
                        gs_binding = torch.from_numpy(point_cloud_data['binding']).long().to(device)
                        my_lbs_weights, my_shapedirs_up = flame_model_new.calculate_interpolated_properties( my_v_cano=gs_xyz, face_indices_for_my_points=gs_binding)
                        my_lbs_weights = my_lbs_weights.cpu().numpy()
                        my_shapedirs_up = my_shapedirs_up.cpu().numpy()
                    print(f"my_lbs_weights shape: {my_lbs_weights.shape}, my_shapedirs_up shape: {my_shapedirs_up.shape}")
                    np.savez(os.path.join(args.output_dir, 'gs_interp.npz'), my_lbs_weights=my_lbs_weights, my_shapedirs_up=my_shapedirs_up) 
                    np.save(os.path.join(args.output_dir, f"new_point_data.npy"), point_cloud_data)
                    tqdm.write(f"Point cloud data saved for sample {i}: {len(xyz_coords)} points")
            

            if batch_idx >= 0: 
                break
    
    print("Test completed!")


if __name__ == "__main__":
    main()
