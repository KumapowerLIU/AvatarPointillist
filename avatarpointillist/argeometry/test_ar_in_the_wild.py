import os
import torch
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
# torch.backends.cuda.matmul.allow_tf32 = True
# 导入你的模块
# from avatarpointillist.argeometry.model.hourglass_cach import HourglassTransformerLM, LMConfig
from avatarpointillist.argeometry.dataset.datasetgpt_test import DataHeadARGPT
def set_seed(seed: int = 42, deterministic: bool = True):
    """
    设置所有相关库的随机种子以保证可复现性。

    参数:
        seed (int): 要设置的种子值。
        deterministic (bool): 是否启用确定性模式（影响 CUDNN）。

    用法:
        set_seed(2025)
    """
    # Python 内置
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡时也设置

 

 

    print(f"[INFO] All seeds set to {seed}, deterministic={deterministic}")

def load_test_model(args, device):
    """加载测试模型"""
    print("Loading models...")

    # 创建Hourglass Transformer模型
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
    
    # # 加载预训练权重
    if args.pretrained_weights:
        print(f"Loading pretrained weights from: {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        
        # 加载模型权重（使用strict=False忽略KV Cache相关参数）
        model.load_state_dict(checkpoint["model"])
        print("Model weights loaded successfully!")
    
    model.to(device)
    model.eval()
    model=torch.compile(model)
    # 加载DINO模型
    dino_model = Dinov2Model.from_pretrained(args.dino_path)
    dino_model.requires_grad_(False)
    dino_model.to(device=device, dtype=torch.float32)
    
    dino_model.eval()
    
    # 加载点云编码器
    point_encoder = load_model(args.point_encoder_ckpt_path)
    point_encoder.requires_grad_(False)
    point_encoder.to(device)
    point_encoder.eval()
    
    # 创建DINO图像处理器
    dino_img_processor = AutoImageProcessor.from_pretrained(args.dino_path)
    
    return model, dino_model, point_encoder, dino_img_processor


def process_condition_data(batch_data, dino_model, point_encoder, device, weight_dtype=torch.float32):
    """处理条件数据"""
    print(333333)
    # 处理DINO特征
    face_dino_imgs = batch_data["dino_imgs"]

    face_features = dino_model(
        face_dino_imgs.to(device)
    ).last_hidden_state
    print(444444)
    # 处理点云特征
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
        # 關鍵：i % group_size 會將 i 映射到 [0, 1, 2, 3] 的循環序列
        position_in_group = i % group_size
        
        # 第一個、第二個、第三個 token 只能是 xyz 類型
        if position_in_group in [0, 1, 2]:
            type2mask[i, :xyz_size] = 0.0
        # 第四個 token 只能是 face 類型
        elif position_in_group == 3:
            type2mask[i, xyz_size:xyz_size + face_nums] = 0.0
            
        # 額外允許結束 token，任何位置都可以是
        type2mask[i, end_idx] = 0.0

    return type2mask
def from_index_to_point(generated_ids, coord_bins=1024, bound_lower=None, bound_upper=None, output_path=None):
    """
    将生成的token序列转换为点云坐标
    
    Args:
        generated_ids: 生成的token序列 (batch_size, seq_len)
        coord_bins: 坐标量化等级，默认512
        bound_lower: 全局最小边界，用于反归一化
        bound_upper: 全局最大边界，用于反归一化
        output_path: 保存PLY文件的路径
    
    Returns:
        xyz_coords: 点云坐标 (N, 3)
        binding_values: 绑定值 (N,)
    """
    from avatarpointillist.argeometry.dataset.data_utils import save_ply
    
    generated_ids = generated_ids.cpu().numpy()
    batch_size, seq_len = generated_ids.shape
    
    # 减去偏移量，恢复原始token范围
    generated_ids = generated_ids 
    
    # 按4个一组分组，舍弃不足4个的token
    num_groups = seq_len // 4
    if num_groups == 0:
        print("Warning: No complete groups found in generated sequence")
        return None, None
    
    # 重塑为 (batch_size, num_groups, 4)
    grouped_ids = generated_ids[:, :num_groups * 4].reshape(batch_size, num_groups, 4)
    
    # 处理每个batch
    all_xyz_coords = []
    all_binding_values = []
    
    for batch_idx in range(batch_size):
        batch_groups = grouped_ids[batch_idx]  # (num_groups, 4)
        
        # 跳过第一组（开始token组）
        if num_groups > 1:
            point_groups = batch_groups[1:]  # 跳过第一组
        else:
            point_groups = batch_groups  # 如果只有一组，也处理
        
        xyz_coords = []
        binding_values = []
        
        for group in point_groups:
            # 每组前3个是xyz坐标，第4个是binding
            x_idx, y_idx, z_idx, binding_idx = group
            
            # 检查是否是EOS token (1-2 = -1)
            if x_idx == -1 or y_idx == -1 or z_idx == -1 or binding_idx == -1:
                break  # 遇到EOS token，停止处理
            
            # 检查坐标是否在有效范围内 [0, coord_bins-1]
            if (0 <= x_idx < coord_bins and 
                0 <= y_idx < coord_bins and 
                0 <= z_idx < coord_bins):
                
                # 将量化坐标转换为[0,1]范围的归一化坐标
                x_norm = x_idx / (coord_bins - 1)
                y_norm = y_idx / (coord_bins - 1)
                z_norm = z_idx / (coord_bins - 1)
                
                xyz_coords.append([x_norm, y_norm, z_norm])
                binding_values.append(binding_idx-  coord_bins)
        
        if len(xyz_coords) > 0:
            xyz_coords = np.array(xyz_coords)
            binding_values = np.array(binding_values)
            
            # 反归一化到原始坐标范围
            if bound_lower is not None and bound_upper is not None:
                xyz_coords = xyz_coords * (bound_upper - bound_lower) + bound_lower
            
            all_xyz_coords.append(xyz_coords)
            all_binding_values.append(binding_values)
            
            print(f"Batch {batch_idx}: Generated {len(xyz_coords)} points")
            
            # 保存PLY文件
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
    """数据批处理函数"""
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
    """主函数"""
    parser = argparse.ArgumentParser('Human AR Model Testing', add_help=False)
    parser.add_argument('--config_dir', type=str, default='/home/liuhongyu/code/AR/aicode/avatarpointillist/configs/humanar_testing.yaml', help='配置文件路径')
    parser.add_argument('--pretrained_weights', type=str, default='/nas5/liuhongyu/AR_Results/checkpoint_gpt/with_mask_norm/checkpoint_40000.pth', help='预训练权重路径')
    # parser.add_argument('--test_data_path', type=str, required=True, help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='./test_outputs_ar', help='输出目录')
    parser.add_argument('--max_length', type=int, default=120000, help='最大生成长度')
    parser.add_argument('--window_size', type=int, default=10000, help='最大生成长度')
    parser.add_argument('--file_path_test', type=str, default=None, help='测试数据路径')

    parser.add_argument('--topk', type=int, default=1, help='窗口大小')
    parser.add_argument('--topp', type=float, default=0.95, help='窗口大小')
    parser.add_argument('--temp', type=float, default=0, help='窗口大小')
    # parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    
    args = parser.parse_args()
    
    # 加载配置文件
    print(f"Loading config from: {args.config_dir}")
    config = OmegaConf.load(args.config_dir)
    
    # 合并命令行参数和配置文件
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
    
    # 转换为Namespace对象
    args = argparse.Namespace(**config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, dino_model, point_encoder, dino_img_processor = load_test_model(args, device)
    
    # 创建测试数据集
    print("Creating test dataset...")
    test_dataset = DataHeadARGPT(args, device=device, test=True)
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batchsize_per_gpu,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of test batches: {len(test_dataloader)}")
    
    # 开始测试
    print("Starting test...")
    model.eval()
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float32):

        for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Processing batches", unit="batch")):

            tqdm.write(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")
            output_path = os.path.join(args.output_dir, f"topk_{str(args.topk)}_topp_{str(args.topp)}_temp_{str(args.temp)}.npy")
            output_path_embe = os.path.join(args.output_dir, f"topk_{str(args.topk)}_topp_{str(args.topp)}_temp_{str(args.temp)}_embed.npy")
            # 处理条件数据
            face_features, pc_feature, shape_latents = process_condition_data(
                batch_data, dino_model, point_encoder, device
            )
            # 生成序列
            
            prompts = batch_data['sequences'][:, :1000].to(device)  # 使用前4个token作为提示
            
            type_mask = get_type_mask(args.max_length,  device)
            generated_ids, out_embedding = model.generate(prompts, args.max_length, temperature=args.temp,  type2mask=type_mask,filter_kwargs={"k": args.topk},
                face_features=face_features, pc_feature=pc_feature, shape_latents= shape_latents 
            )
            # generated_ids = batch_data['sequences'].to(device)
            
            # 保存原始token序列
            np.save(output_path, generated_ids.cpu().numpy())
            np.save(output_path_embe, out_embedding.cpu().numpy())
            tqdm.write(f"Generated sequence saved to: {output_path}")
            tqdm.write(f"Generated sequence shape: {generated_ids.shape}")
            tqdm.write(f"Generated emb shape: {out_embedding.shape}")

            tqdm.write(f"Generated sequence length: {generated_ids.size(1)}")
            
            # 转换为点云坐标
            tqdm.write("Converting tokens to point cloud...")
            xyz_coords_list, binding_values_list = from_index_to_point(
                generated_ids,
                coord_bins=args.n_discrete_size,
                bound_lower=np.array(args.global_min, dtype=np.float32),
                bound_upper=np.array(args.global_max, dtype=np.float32),
                output_path=os.path.join(args.output_dir, f"topk_{str(args.topk)}_topp_{str(args.topp)}_temp_{str(args.temp)}")
            )
            
            # 保存点云数据
            for i, (xyz_coords, binding_values) in enumerate(zip(xyz_coords_list, binding_values_list)):
                if len(xyz_coords) > 0:
                    point_cloud_data = {
                        'xyz': xyz_coords,
                        'binding': binding_values
                    } 
                    np.save(os.path.join(args.output_dir, f"point_cloud_data_batch_topk_{str(args.topk)}_topp_{str(args.topp)}_temp_{str(args.temp)}.npy"), point_cloud_data)
                    tqdm.write(f"Point cloud data saved for sample {i}: {len(xyz_coords)} points")
            
            # 只测试前几个batch
            if batch_idx >= 0:  # 测试前3个batch
                break
    
    print("Test completed!")


if __name__ == "__main__":
    main()
