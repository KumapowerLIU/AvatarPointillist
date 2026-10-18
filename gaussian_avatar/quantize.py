import numpy as np
from plyfile import PlyData, PlyElement
from torch import nn
import torch

from flame_model.flame import FlameHead

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


scaling_activation = torch.exp
scaling_inverse_activation = torch.log

opacity_activation = torch.sigmoid
inverse_opacity_activation = inverse_sigmoid


def quantize_coords_torch(coords, num_bins=1024, coord_range=(0.0, 1.0)):
    """
    将连续坐标量化为整数 token（bin index）

    Args:
        coords: torch.Tensor of shape (N, 3), float32
        num_bins: int, 量化等级（默认 1024）
        coord_range: tuple, 坐标的最小值和最大值（默认范围为 [0.0, 1.0]）

    Returns:
        quantized: torch.Tensor of shape (N, 3), int64, 每个坐标轴是一个整数 token [0, num_bins-1]
    """
    min_val, max_val = coord_range
    coords_clipped = torch.clamp(coords, min=min_val, max=max_val)
    scaled = (coords_clipped - min_val) / (max_val - min_val)
    quantized = torch.floor(scaled * (num_bins - 1)).long()
    return quantized

def compare_before_after(name, original, dequantized):
    # 归一化反量化后的向量
    dequantized_normalized = torch.nn.functional.normalize(dequantized, p=2, dim=1)
    norm = torch.norm(original, p=2, dim=1, keepdim=True)
    recovered = dequantized_normalized * norm

    # 误差
    l2_error = torch.norm(original - recovered, dim=1).mean()
    max_error = torch.abs(original - recovered).max()
    print(f"{name} Dequantization Restore - L2 Mean Error: {l2_error:.6f}, Max Abs Error: {max_error:.6f}")
def dequantize_coords_torch(quantized, num_bins=1024, coord_range=(0.0, 1.0)):
    """
    将量化后的 token 还原回坐标

    Args:
        quantized: torch.Tensor of shape (N, 3), int64
        num_bins: int, 量化等级
        coord_range: tuple, 坐标范围

    Returns:
        coords: torch.Tensor of shape (N, 3), float32
    """
    min_val, max_val = coord_range
    scaled = quantized.float() / (num_bins - 1)
    coords = scaled * (max_val - min_val) + min_val
    return coords

def print_error_stats(name, original, recovered):
    l2_error = torch.norm(original - recovered, dim=1).mean()
    max_error = torch.abs(original - recovered).max()
    print(f"{name} - L2 Mean Error: {l2_error:.6f}, Max Abs Error: {max_error:.6f}")

def quantize_gs(xyz, opacities, rgb, scales, rots):
    xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
    rgb = nn.Parameter(torch.tensor(rgb, dtype=torch.float, device="cuda").requires_grad_(False))
    opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
    scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
    rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))


    print(xyz.shape)

    print(rgb.shape)
    print(opacity.shape)
    print(scaling.shape)
    print(rotation.shape)

    # torch.Size([95231, 3])
    # torch.Size([95231, 3])
    # torch.Size([95231, 1])
    # torch.Size([95231, 3])
    # torch.Size([95231, 4])

    xyz_norm = torch.norm(xyz, p=2, dim=1, keepdim=True)
    xyz_normalized = torch.nn.functional.normalize(xyz)

    rgb_norm = torch.norm(rgb, p=2, dim=1, keepdim=True)
    rgb_normalized = torch.nn.functional.normalize(rgb)

    opacity_norm = torch.norm(opacity, p=2, dim=1, keepdim=True)
    opacity_normalized = torch.nn.functional.normalize(opacity)

    scaling_norm = torch.norm(scaling, p=2, dim=1, keepdim=True)
    scaling_normalized = torch.nn.functional.normalize(scaling)

    rotation_norm = torch.norm(rotation, p=2, dim=1, keepdim=True)
    rotation_normalized = torch.nn.functional.normalize(rotation)

    analysis_torch(scaling, 'scaling_original')

    analysis_torch(xyz_normalized, 'xyz')
    analysis_torch(rgb_normalized, 'rgb')
    analysis_torch(opacity_normalized, 'opacity')
    analysis_torch(scaling_normalized, 'scaling')
    analysis_torch(rotation_normalized, 'rot')

    # 每个属性使用的 token 数量（例如 1024 个 bin）
    bins_per_feature = 1024

    # 各属性索引偏移（手动分配码本段）
    xyz_offset = 0
    rgb_offset = xyz_offset + bins_per_feature * 1
    opacity_offset = rgb_offset + bins_per_feature * 1
    scaling_offset = opacity_offset + bins_per_feature * 1
    rotation_offset = scaling_offset + bins_per_feature * 1

    xyz_token = quantize_coords_torch(xyz_normalized, num_bins=bins_per_feature, coord_range=(-1.0, 1.0)) + xyz_offset
    rgb_token = quantize_coords_torch(rgb_normalized, num_bins=bins_per_feature, coord_range=(-1.0, 1.0)) + rgb_offset
    opacity_token = quantize_coords_torch(opacity_normalized, num_bins=bins_per_feature,
                                          coord_range=(-1.0, 1.0)) + opacity_offset
    scaling_token = quantize_coords_torch(scaling_normalized, num_bins=bins_per_feature,
                                          coord_range=(-1.0, 0)) + scaling_offset
    rotation_token = quantize_coords_torch(rotation_normalized, num_bins=bins_per_feature,
                                           coord_range=(-1.0, 1.0)) + rotation_offset

    xyz_token_raw = xyz_token - xyz_offset
    rgb_token_raw = rgb_token - rgb_offset
    opacity_token_raw = opacity_token - opacity_offset
    scaling_token_raw = scaling_token - scaling_offset
    rotation_token_raw = rotation_token - rotation_offset

    xyz_dequantized = dequantize_coords_torch(xyz_token_raw, num_bins=bins_per_feature, coord_range=(-1.0, 1.0))
    rgb_dequantized = dequantize_coords_torch(rgb_token_raw, num_bins=bins_per_feature, coord_range=(-1.0, 1.0))
    opacity_dequantized = dequantize_coords_torch(opacity_token_raw, num_bins=bins_per_feature, coord_range=(-1.0, 1.0))
    scaling_dequantized = dequantize_coords_torch(scaling_token_raw, num_bins=bins_per_feature, coord_range=(-1.0, 0))
    rotation_dequantized = dequantize_coords_torch(rotation_token_raw, num_bins=bins_per_feature,
                                                   coord_range=(-1.0, 1.0))

    print_error_stats("XYZ", xyz_normalized, xyz_dequantized)
    print_error_stats("RGB", rgb_normalized, rgb_dequantized)
    print_error_stats("Opacity", opacity_normalized, opacity_dequantized)
    print_error_stats("Scaling", scaling_normalized, scaling_dequantized)
    print_error_stats("Rotation", rotation_normalized, rotation_dequantized)
    print(111111111111111111111111)
    compare_before_after("XYZ", xyz, xyz_normalized)
    compare_before_after("RGB", rgb, rgb_normalized)
    compare_before_after("Opacity", opacity, opacity_normalized)
    compare_before_after("Scaling", scaling, scaling_normalized)
    compare_before_after("Rotation", rotation, rotation_normalized)
    print(22222222222222222222222222)

    compare_before_after("XYZ", xyz, xyz_dequantized)
    compare_before_after("RGB", rgb, rgb_dequantized)
    compare_before_after("Opacity", opacity, opacity_dequantized)
    compare_before_after("Scaling", scaling, scaling_dequantized)
    compare_before_after("Rotation", rotation, rotation_dequantized)


def analysis_torch(input_feature: torch.Tensor, feature_name: str):
    """
    对输入特征张量按列统计 max, min, mean, var 并打印结果。

    Args:
        input_feature: torch.Tensor of shape (N, D)
        feature_name: str, 特征名前缀
    """
    # 统计各列（维度）上的统计量
    max_vals = torch.max(input_feature, dim=0).values
    min_vals = torch.min(input_feature, dim=0).values
    mean_vals = torch.mean(input_feature, dim=0)
    var_vals = torch.var(input_feature, dim=0, unbiased=False)  # 与 numpy 保持一致，除以 N

    # 打印结果
    for i in range(input_feature.shape[1]):
        print(f"{feature_name}_{i}:")
        print(f"  Max:  {max_vals[i].item():.4f}")
        print(f"  Min:  {min_vals[i].item():.4f}")
        print(f"  Mean: {mean_vals[i].item():.4f}")
        print(f"  Var:  {var_vals[i].item():.6f}")


def ply_load(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    rgb = np.stack((np.asarray(plydata.elements[0]["rgb_0"]),
                    np.asarray(plydata.elements[0]["rgb_1"]),
                    np.asarray(plydata.elements[0]["rgb_2"])), axis=1)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
    if len(binding_names) > 0:
        binding_names = sorted(binding_names, key=lambda x: int(x.split('_')[-1]))
        binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
        for idx, attr_name in enumerate(binding_names):
            binding[:, idx] = np.asarray(plydata.elements[0][attr_name])
    return xyz, opacities, rgb, scales, rots, binding


def count_occurrence_distribution(data):
    """
    统计数组中每种出现次数对应的数字个数

    Args:
        data (np.ndarray): 输入数组

    Returns:
        dict: {出现次数: 数字的个数}
    """
    # 使用 np.unique 统计每个数字的出现次数
    _, counts = np.unique(data, return_counts=True)

    # 使用 np.unique 再次统计每种出现次数的分布
    unique_counts, frequency = np.unique(counts, return_counts=True)

    # 返回结果为字典 {出现次数: 数字的个数}
    return dict(zip(unique_counts, frequency))
# 示例使用
if __name__ == "__main__":
    xyz, opacities, rgb, scales, rots, binding = ply_load(
        '/nas5/liuhongyu/AR_Results/GSA_RGB/output/UNION10_074_EMO1234EXP234589_eval_600k_nooffset_max5min1_newblind_useoffset/point_cloud/iteration_600000/point_cloud.ply')

    print(xyz.shape)
    print( binding.shape)
    # n_shape = 300
    # n_expr = 100
    # flame_model = FlameHead(
    #     n_shape,
    #     n_expr,
    #     add_teeth=True,
    # ).cuda()
    # print(flame_model.faces.shape[0])
    # print(xyz.shape)
    # print(binding.shape)
    # unique, counts = np.unique(binding, return_counts=True)
    # print(unique)
    # print(counts)
    # total = np.sum( counts)
    # print(counts.max())
    # print(counts.min())
    #
    # # 输出结果
    # print("总和（unique × counts）:", total)
    #
    #
    #
    # # # 将数字和对应的计数配对
    # num_count_pairs = np.array(list(zip(unique, counts)))
    #
    # # 按照出现次数排序（降序）
    # num_count_pairs_sorted = num_count_pairs[num_count_pairs[:, 1].argsort()][::-1]
    #
    # # 获取重复最多的前 10 个数字及其出现次数
    # top_10 = num_count_pairs_sorted[:10]
    #
    # # 打印结果
    # print("重复最多的10个数字及其出现次数:")
    # for num, count in top_10:
    #     print(f"数字: {int(num)}, 出现次数: {int(count)}")
    # result = count_occurrence_distribution(binding)
    # #
    # # # 打印结果
    # print("出现次数统计分布:")
    # for occurrence, num_count in result.items():
    #     print(f"出现次数为 {occurrence} 的数字有 {num_count} 个")
    # print(binding)
    # print(binding.shape)
    # quantize_gs(xyz, opacities, rgb, scales, rots)

    # 假设有 4 个顶点，每个是 3D 坐标
    # vertices = np.array([
    #     [0.1, 0.2, 0.3],
    #     [0.9, 0.8, 0.7],
    #     [0.5, 0.5, 0.5],
    #     [0.0, 1.0, 0.0]
    # ], dtype=np.float32)
    #
    # # 量化
    # quantized = quantize_coords(vertices, num_bins=1024)
    # print("Quantized tokens:\n", quantized)
    #
    # # 反量化
    # recovered = dequantize_coords(quantized, num_bins=1024)
    # print("Recovered coordinates:\n", recovered)
