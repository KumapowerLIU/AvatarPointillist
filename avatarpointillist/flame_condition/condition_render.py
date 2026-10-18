# render the target face of mesh with orthogonal projection

import tyro
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import time
import numpy as np
import torch
import PIL.Image
from PIL import Image
import torch.nn.functional as F
import os
from avatarpointillist.flame_condition.flame import FlameHead

def barycentric_coordinates_center(p, a, b, c):
    """计算点 p 相对于三角形 (a, b, c) 的重心坐标。"""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = torch.einsum('...i,...i->...', v0, v0)
    d01 = torch.einsum('...i,...i->...', v0, v1)
    d11 = torch.einsum('...i,...i->...', v1, v1)
    d20 = torch.einsum('...i,...i->...', v2, v0)
    d21 = torch.einsum('...i,...i->...', v2, v1)
    denom = d00 * d11 - d01 * d01
    # 为防止退化三角形导致除以零，增加一个极小值
    v = (d11 * d20 - d01 * d21) / (denom + 1e-8)
    w = (d00 * d21 - d01 * d20) / (denom + 1e-8)
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


def inside_triangle_center(p, a, b, c):
    """使用叉乘判断点 p 是否在三角形 (a, b, c) 内部。"""
    c1 = (b - a).unsqueeze(-2)
    c2 = (c - b).unsqueeze(-2)
    c3 = (a - c).unsqueeze(-2)
    p1 = p - a.unsqueeze(-2)
    p2 = p - b.unsqueeze(-2)
    p3 = p - c.unsqueeze(-2)

    z1 = c1[..., 0] * p1[..., 1] - c1[..., 1] * p1[..., 0]
    z2 = c2[..., 0] * p2[..., 1] - c2[..., 1] * p2[..., 0]
    z3 = c3[..., 0] * p3[..., 1] - c3[..., 1] * p3[..., 0]

    return (z1 >= 0) & (z2 >= 0) & (z3 >= 0) | (z1 <= 0) & (z2 <= 0) & (z3 <= 0)


def save_tensor_as_png(image_tensor: torch.Tensor ):
    """
    将一个形状为 (H, W, C)，范围在 [0, 1] 的浮点型 Tensor 保存为 PNG 图像。

    Args:
        image_tensor (torch.Tensor): 输入的图像张量。
        output_path (str): 输出的 PNG 文件路径 (例如: 'output/face_texture.png')。
    """
    # --- 参数校验 ---
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"输入必须是 torch.Tensor, 但收到了 {type(image_tensor)}")
    if image_tensor.dim() != 3 or image_tensor.shape[2] != 3:
        raise ValueError(f"期望的张量形状是 (H, W, 3)，但收到了 {image_tensor.shape}")
    if image_tensor.max() > 1.0 or image_tensor.min() < 0.0:
        print("警告: 输入张量的数值范围超出了 [0, 1]，可能会导致图像裁剪失真。")

    # --- 核心转换步骤 ---

    # 1. 将数值范围从 [0.0, 1.0] 转换到 [0, 255]
    #    并转换为 8 位无符号整数 (uint8)，这是图像的标准格式。
    image_to_save = (image_tensor.clamp(0, 1) * 255).byte()

    # 2. 如果张量在 GPU 上，需要先移动到 CPU
    if image_to_save.is_cuda:
        image_to_save = image_to_save.cpu()

    # 3. 将 PyTorch Tensor 转换为 NumPy 数组
    #    Pillow 库可以直接处理 NumPy 数组
    numpy_array = image_to_save.numpy()

    # 4. 从 NumPy 数组创建 PIL Image 对象
    pil_image = Image.fromarray(numpy_array, 'RGB')

    return pil_image


class FlameViewerFitting:
    def __init__(self, device, n_shape=300, n_expr=100):

        # flame model
        self.flame_model = FlameHead(n_shape, n_expr, add_teeth=True).to(device)
        self.device = device
        # flame settings
        # self.flame_param = dict(np.load(param_path))
        # for k, v in self.flame_param.items():
        #     if v.dtype in [np.float64, np.float32]:
        #         self.flame_param[k] = torch.from_numpy(v).float()
        # self.forward_flame(self.flame_param)
    def get_flame_param_bound(self, param_path):
        with torch.no_grad():
            flame_param = dict(np.load(param_path))
            for k, v in flame_param.items():
                if v.dtype in [np.float64, np.float32]:
                    flame_param[k] = torch.from_numpy(v).float()
            verts, verts_cano = self.flame_model(
                flame_param['shape'][None, ...].to(self.device),
                torch.zeros_like(flame_param['expr'][[0]]).to(self.device),
                torch.zeros_like(flame_param['rotation'][[0]]).to(self.device),
                torch.zeros_like(flame_param['neck_pose'][[0]]).to(self.device),
                torch.zeros_like(flame_param['jaw_pose'][[0]]).to(self.device),
                torch.zeros_like(flame_param['eyes_pose'][[0]]).to(self.device),
                torch.zeros_like(flame_param['translation'][[0]]).to(self.device),
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=None,
                dynamic_offset=None,
            )
            bound_upper = torch.max(verts_cano[0].detach(), dim = 0)[0] + 0.005
            bound_lower = torch.min(verts_cano[0].detach(), dim = 0)[0] - 0.005
            trip_spatial_bounds = torch.cat([bound_lower, bound_upper], dim = 0).reshape((2, 3))
            trip_spatial_bounds[0, 1] -= 0.025
            trip_spatial_bounds[0, 0] = -0.25
            trip_spatial_bounds[1, 0] = 0.25
            return trip_spatial_bounds
    def forward_flame(self, flame_param):
        N = flame_param['expr'].shape[0]
        with torch.no_grad():
            self.verts, self.verts_cano = self.flame_model(
                flame_param['shape'][None, ...].expand(N, -1).cuda(),
                torch.zeros_like(flame_param['expr'].cuda()),
                flame_param['rotation'].cuda(),
                flame_param['neck_pose'].cuda(),
                flame_param['jaw_pose'].cuda(),
                flame_param['eyes_pose'].cuda(),
                flame_param['translation'].cuda(),
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=flame_param['static_offset'].cuda(),
                # dynamic_offset=flame_param['dynamic_offset'].cuda(),
            )

    def get_face_texture_tensor_centered(
            self,
            face_index: int,
            verts: torch.Tensor,
            faces: torch.Tensor,
            verts_uv: torch.Tensor,
            faces_uv: torch.Tensor,
            texture: torch.Tensor,
            resolution: int = 256
    ) -> torch.Tensor:
        """
        提取单个面片的纹理，保持其原始宽高比，并在输出图像中居中。
        """
        device = verts.device

        # 1. 获取指定面片的顶点和UV坐标
        face_vert_indices = faces[face_index]
        v1, v2, v3 = verts[face_vert_indices]

        face_uv_indices = faces_uv[face_index]
        uv1, uv2, uv3 = verts_uv[face_uv_indices]

        v1_2d, v2_2d, v3_2d = v1[:2], v2[:2], v3[:2]
        triangle_2d = torch.stack([v1_2d, v2_2d, v3_2d])

        # 2. 为输出图像创建一个 [0,1] 范围的像素坐标网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, resolution, device=device),
            torch.linspace(0, 1, resolution, device=device),
            indexing='ij'
        )
        grid_normalized = torch.stack([grid_x, grid_y], dim=-1)

        # ------------------- 新的归一化与居中逻辑 -------------------
        # 目标：将 [0,1] 的网格映射回三角形的原始世界空间，同时确保三角形居中且不失真。

        # a. 找到三角形在2D世界空间中的包围盒
        min_xy = torch.min(triangle_2d, dim=0).values
        max_xy = torch.max(triangle_2d, dim=0).values
        box_wh = max_xy - min_xy  # 包围盒的宽高

        # b. 找到包围盒的最长边
        box_max_dim = torch.max(box_wh)
        if box_max_dim < 1e-6:  # 处理退化三角形（面积为0）的情况
            return torch.zeros((resolution, resolution, texture.shape[1]), dtype=torch.float32, device=device)

        # c. 计算统一的缩放因子
        scale = box_max_dim

        # d. 计算将缩放后的包围盒居中所需的偏移量
        #    新的缩放后尺寸为 box_wh / scale。
        #    边距 = (1 - 新尺寸) / 2。
        #    这个偏移量需要在变换前从 min_xy 中减去。
        offset = (scale - box_wh) / 2.0

        # e. 创建逆变换：从归一化网格坐标 -> 世界坐标
        #    一个世界坐标点 p 被归一化为：(p - min_xy + offset) / scale
        #    所以逆变换，从归一化到世界坐标为：(normalized * scale) - offset + min_xy
        grid_world = grid_normalized * scale - offset + min_xy
        # ----------------- 结束新的归一化与居中逻辑 -----------------

        # 3. 判断哪些像素点在三角形内部
        mask = inside_triangle_center(grid_world, v1_2d, v2_2d, v3_2d)

        # 4. 对内部的像素点，计算其重心坐标
        bary_coords = barycentric_coordinates_center(grid_world, v1_2d, v2_2d, v3_2d)

        # 5. 使用重心坐标插值计算出每个内部像素点对应的UV坐标
        uv_coords_interpolated = torch.einsum('hw,c->hwc', bary_coords[:, :, 0], uv1) + \
                                 torch.einsum('hw,c->hwc', bary_coords[:, :, 1], uv2) + \
                                 torch.einsum('hw,c->hwc', bary_coords[:, :, 2], uv3)

        # 6. 使用插值后的UV坐标去纹理图上采样
        # grid_sample 期望坐标在 [-1, 1] 范围内，所以我们进行变换
        uv_for_sampling = uv_coords_interpolated * 2.0 - 1.0

        texture_batch = texture
        uv_grid_batch = uv_for_sampling.unsqueeze(0)

        sampled_colors = F.grid_sample(
            texture_batch,
            uv_grid_batch,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # 7. 创建最终图像
        image = torch.zeros((resolution, resolution, texture.shape[1]), dtype=torch.float32, device=device)
        image[mask] = sampled_colors[mask]

        return image

    def get_face_feature(self, face_index, dinofeature_tex, resolution=256):
        with torch.no_grad():

            verts_uv = self.flame_model.verts_uvs.clone()
            verts_uv[:, 1] = 1 - verts_uv[:, 1]

            faces_uv = self.flame_model.textures_idx.int()
            verts = self.verts_cano[[0]]
            faces = self.flame_model.faces
            tex = dinofeature_tex.to(verts.device)
            # since the batch size of verts is 1
            triangle_texture = self.get_face_texture_tensor_centered(face_index, verts[0], faces, verts_uv, faces_uv,
                                                                      tex, resolution=resolution)
            return save_tensor_as_png(triangle_texture)


# if __name__ == "__main__":
#     render_class = FlameViewer(cfg)
#     render_class.run_single_face(3763)
