# import numpy as np

# a = np.load('/nas5/liuhongyu/AR_Results/baseline/ours/vfhq/0001/ar_out/topk_1_topp_0.95_temp_0.0_embed.npy')
# print(a.shape)
# ar_feaures = a.squeeze(0)[3:, :]
# seq_len = ar_feaures.shape[0] // 4
# print(seq_len)
# k =  np.load('/nas5/liuhongyu/AR_Results/ners_test_data/ar_out/ar_out/093/new_point_data.npy',   allow_pickle=True).item()
# xyz = k['xyz']
# print(xyz.shape)


# import os
# import cv2

# # 设置视频所在的目录和输出目录
# video_dir = '/nas5/liuhongyu/AR_Results/baseline/ours/vfhq/HUNYUAN/'  # ← 修改为你的视频目录
# output_root = '/nas5/liuhongyu/AR_Results/baseline/ours/vfhq/HUNYUAN/gen_frames'  # ← 修改为帧保存目录

# # 支持的视频扩展名
# video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

# # 创建输出根目录（如果不存在）
# os.makedirs(output_root, exist_ok=True)

# # 遍历视频目录下的所有文件
# for filename in os.listdir(video_dir):
#     file_path = os.path.join(video_dir, filename)
    
#     # 检查是否是视频文件
#     if not any(filename.lower().endswith(ext) for ext in video_extensions):
#         continue

#     # 创建每个视频对应的输出文件夹
#     video_name = os.path.splitext(filename)[0]
#     output_dir = os.path.join(output_root, video_name)
#     os.makedirs(output_dir, exist_ok=True)

#     print(f"Processing video: {filename}")
#     cap = cv2.VideoCapture(file_path)
#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # 视频读取完毕

#         # 保存当前帧为图片
#         frame_filename = f"frame_{frame_idx:04d}.jpg"
#         frame_path = os.path.join(output_dir, frame_filename)
#         cv2.imwrite(frame_path, frame)
#         frame_idx += 1

#     cap.release()
#     print(f"Saved {frame_idx} frames to {output_dir}")

# print("✅ 所有视频处理完成！")


# import matplotlib.pyplot as plt
# import numpy as np

# # 设置绘图风格
# plt.style.use('seaborn-v0_8-muted')
# np.random.seed(42)

# def generate_improved_avatar_plot():
#     n_ids = 400
#     id_positions = np.linspace(5, 95, n_ids)
    
#     # 模拟每个ID天然的微小差异 (Natural Variance)
#     # 即使是原始数据，由于点数和分布不同，在纵轴上也有微小摆动
#     natural_variance = np.random.normal(0, 2, n_ids)
    
#     fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
#     colors = ['#1f77b4', '#ff7f0e', '#d62728']

#     # --- Stage 1: Raw Data ---
#     # 体现你说的：分布不太一样，但区别不剧烈
#     axes[0].scatter(id_positions, natural_variance, alpha=0.8, s=25, c=colors[0], label='Unique Identities')
#     axes[0].set_title("Stage 1: Raw Data\n(Natural Geometric Variation)", fontsize=16, pad=15)
#     axes[0].text(5, 40, "Initial Diversity: Low", color='gray', fontsize=12)

#     # --- Stage 2: Random Window Sampling ---
#     # 在天然差异的基础上，由于2500/25000的随机采样，变体剧增
#     window_samples_x = []
#     window_samples_y = []
#     for i, pos in enumerate(id_positions):
#         # 基于该ID的初始位置(natural_variance[i])进行扩散
#         variations = natural_variance[i] + np.random.normal(0, 8, 15)
#         window_samples_x.extend([pos] * 15)
#         window_samples_y.extend(variations)
    
#     axes[1].scatter(window_samples_x, window_samples_y, alpha=0.3, s=8, c=colors[1], label='Sampled Windows')
#     axes[1].set_title("Stage 2: Window Sampling\n(Combinatorial Combinations)", fontsize=16, pad=15)

#     # --- Stage 3: Forgetful Masking ---
#     # 叠加掩码策略，分布彻底铺开
#     masked_samples_x = []
#     masked_samples_y = []
#     for i, pos in enumerate(id_positions):
#         variations = natural_variance[i] + np.random.normal(0, 18, 50) 
#         masked_samples_x.extend([pos] * 50)
#         masked_samples_y.extend(variations)

#     axes[2].scatter(masked_samples_x, masked_samples_y, alpha=0.1, s=4, c=colors[2], label='Masked Variations')
#     axes[2].set_title("Stage 3: Forgetful Masking\n(Robust Feature Manifold)", fontsize=16, pad=15)

#     # 统一格式
#     for i, ax in enumerate(axes):
#         ax.set_xlabel("Inter-ID Semantic Space", fontsize=13)
#         if i == 0:
#             ax.set_ylabel("Intra-ID Geometric Variation", fontsize=13)
#         ax.set_ylim(-50, 50)
#         ax.set_xlim(0, 100)
#         ax.grid(True, linestyle='--', alpha=0.3)
#         ax.legend(loc='lower right')

#     plt.tight_layout()
#     plt.savefig('avatar_distribution_v2.png', dpi=300)
#     plt.show()

# generate_improved_avatar_plot()




import torch
import numpy as np
import os
from plyfile import PlyData, PlyElement
from avatarpointillist.argeometry.dataset.data_utils import ConsistentPointSampler, compute_face_orientation,save_ply
from avatarpointillist.flame_condition.flame import FlameHeadSubdivided, FlameHead
def quantize_coords_numpy(coords, num_bins=1024, coord_range=(0.0, 1.0)):
    """
    Quantize continuous coordinates to integer tokens (bin index)
    x-axis: [0, num_bins-1]
    y-axis: [num_bins, 2*num_bins-1] 
    z-axis: [2*num_bins, 3*num_bins-1]

    Args:
        coords: np.ndarray of shape (N, 3), float32
        num_bins: int, quantization levels per axis (default 1024)
        coord_range: tuple, min and max values of coordinates (default range [0.0, 1.0])

    Returns:
        quantized: np.ndarray of shape (N, 3), int64, each axis mapped to different bin ranges
    """
    min_val, max_val = coord_range
    # Clip coordinates to range
    coords_clipped = np.clip(coords, min_val, max_val)
    # Scale coordinates to [0, 1]
    scaled = (coords_clipped - min_val) / (max_val - min_val)
    # Quantize scaled coordinates to integers
    quantized = np.floor(scaled * (num_bins - 1)).astype(np.int64)
    
    return quantized
bound_upper = np.array( [-0.37134886, -0.26413554, -0.3435335], dtype=np.float32)
bound_lower = np.array([0.30597788, 0.21042103, 0.12734485], dtype=np.float32)
coord_bins = 1024
device = 'cuda'
flame_base_path = '/nas5/liuhongyu/AR_Results/cvpr2026_code_review/toy_data/flame/flame_assets/flame'
flame_model = FlameHead(
    300,
    100,
    flame_model_path=f'{flame_base_path}/flame2023.pkl',
    flame_lmk_embedding_path=f"{flame_base_path}/landmark_embedding_with_eyes.npy",
    flame_template_mesh_path=f"{flame_base_path}/head_template_mesh.obj",
    flame_mask_path=f"{flame_base_path}/FLAME_masks.pkl",

) 
flame_model = flame_model.cuda()       
def get_cano_xyz( xyz, flame_param, binding):
    flame_param = np.load(flame_param)
    verts, verts_cano = flame_model(
        torch.from_numpy(flame_param['shape'])[None, ...].to(device),
        torch.zeros(flame_param['expr'][[0]].shape).to(device),
        torch.zeros(flame_param['rotation'][[0]].shape).to(device),
        torch.zeros(flame_param['neck_pose'][[0]].shape).to(device),
        torch.zeros(flame_param['jaw_pose'][[0]].shape).to(device),
        torch.zeros(flame_param['eyes_pose'][[0]].shape).to(device),
        torch.zeros(flame_param['translation'][[0]].shape).to(device),
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        static_offset=None,
        dynamic_offset=None,
    )
    face_center_cano = verts_cano[:, flame_model.faces].mean(dim=-2).squeeze(0)
    face_orien_mat_cano, face_scaling_cano = compute_face_orientation(verts_cano.squeeze(0), flame_model.faces.squeeze(0), return_scale=True)

    xyz_cano = torch.bmm(face_orien_mat_cano[binding], xyz[..., None]).squeeze(-1)
    xyz_cano = xyz_cano * face_scaling_cano[binding] + face_center_cano[binding]
    xyz_cano  = xyz_cano.detach().cpu().numpy()
    return xyz_cano 

def sort_point_cloud(  xyz, binding):
    """
    对点云进行排序：从下往上，从左往右
    同时保留原始索引以查找对应的binding值和feature值
    
    Args:
        xyz: np.ndarray of shape (N, 3), 原始点云坐标
        binding: np.ndarray of shape (N, M), 每个点对应的binding值
        triplane: TriPlaneNetwork, 用于提取特征的三平面网络
        
    Returns:
        sorted_sequence: np.ndarray, 排序后的序列，包含量化的xyz和binding
        original_indices: np.ndarray, 排序后的原始索引
        sorted_features: np.ndarray, 排序后的特征，与序列一一对应（CPU上的numpy数组）
    """
    # 1. 使用spatial_bounds归一化坐标到[0,1]范围
    # xyz_min = np.min(xyz, axis=0)
    # xyz_max = np.max(xyz, axis=0)
    xyz_norm = (xyz -  bound_lower) / ( bound_upper - bound_lower)
    # xyz_norm = xyz
    # xyz_norm = (xyz - self.spatial_bounds[0]) / (self.spatial_bounds[1] - self.spatial_bounds[0]) 
    # 2. 量化坐标到整数范围 [0, coord_bins-1]
    xyz_quantized = quantize_coords_numpy(xyz_norm,  coord_bins, (0.0, 1.0))
    
    # 3. 创建包含原始索引的数组
    indices_with_coords = np.column_stack([xyz_quantized, np.arange(len(xyz))])
    
    # 4. 排序：从下往上(y)，从左往右(x)，从前到后(z)
    # 使用字典序排序：y, x, z (优先级：y > x > z)
    sort_indices = np.lexsort((indices_with_coords[:, 2], indices_with_coords[:, 0], indices_with_coords[:, 1]))
    
    # 5. 应用排序
    sorted_coords = xyz_quantized[sort_indices]
    original_indices = indices_with_coords[sort_indices, 3].astype(int)
    sorted_binding = binding[original_indices]
    
    # 6. 构建序列：每个点包含 [x, y, z, binding+offset]
    # binding值加上offset避免与坐标token重叠
    # 如果binding是多维的，取第一个值；如果是一维的，直接使用
    if len(sorted_binding.shape) > 1:
        binding_values = sorted_binding[:, 0]  # 取第一列
    else:
        binding_values = sorted_binding
    # xyz and binding token offset
    binding_with_offset = binding_values + coord_bins
    
    # 7. 创建序列，每个点4个token
    sequence = []
    for i in range(len(sorted_coords)):
        point_tokens = [sorted_coords[i, 0], sorted_coords[i, 1], sorted_coords[i, 2], binding_with_offset[i]]
        sequence.extend(point_tokens)

    
    return np.array(sequence) 
def ply_load(  path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
    if len(binding_names) > 0:
        binding_names = sorted(binding_names, key=lambda x: int(x.split('_')[-1]))
        binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
        for idx, attr_name in enumerate(binding_names):
            binding[:, idx] = np.asarray(plydata.elements[0][attr_name])
    return xyz, binding

init_path = '/nas5/liuhongyu/AR_Results/clusterbackup/GSAFitting'
all_ids = os.listdir(init_path)
all_ids.sort()
sequence_sum = 0
count = 0 
for id in all_ids:
    ply_path = os.path.join(init_path, id, 'point_cloud', 'iteration_600000', 'point_cloud.ply')
    flame_param_path = os.path.join(init_path, id, 'point_cloud', 'iteration_600000', 'flame_param.npz')

    xyz, binding = ply_load(ply_path)
    xyz_cano = get_cano_xyz(torch.tensor(xyz, device="cuda").float(), flame_param_path, torch.tensor(binding, device="cuda").long().squeeze(-1))
    sequence = sort_point_cloud(xyz_cano, binding)
    sequence = sequence[0:996]   
    sequence_sum = sequence_sum + sequence

    count += 1
sequence_mean = sequence_sum / count
sequence_mean_floor = np.floor(sequence_mean)
np.save("sequence_mean_floor.npy", sequence_mean_floor)


print(sequence_mean_floor )
        
