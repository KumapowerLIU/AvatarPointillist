import torch
import torch.nn as nn
from avatarpointillist.gstransformer.models.rendering.flame_model.flame_GSA  import FlameHead
import numpy as np
from plyfile import PlyData
from avatarpointillist.argeometry.dataset.data_utils import ConsistentPointSampler, compute_face_orientation,save_ply
from plyfile import PlyData, PlyElement

def save_ply(path, xyz, bind):
    l = ['x', 'y', 'z']
    for i in range(1):
        l.append('binding_{}'.format(i))
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes =  xyz

    attributes = np.concatenate((attributes, bind[:, None]), axis=1)

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)



def jitter_point_cloud(points: np.ndarray, sigma: float = 0.005, clip: float = 0.01) -> np.ndarray:
    """
    对3D点云进行随机抖动（数据增强）。

    Args:
        points (np.ndarray): 输入的点云数组，形状为 (N, 3)。
        sigma (float): 高斯噪声的标准差，控制抖动的基础幅度。
        clip (float): 噪声的最大裁剪值（绝对值），防止出现极端离群点。

    Returns:
        np.ndarray: 增加了噪声后的点云数组，形状仍为 (N, 3)。
    """
    # 检查输入是否为numpy数组
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # 检查维度是否正确
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("输入点云的形状必须是 (N, 3)")

    # 生成与点云形状相同的高斯噪声
    # loc=0 表示噪声均值为0，scale=sigma 表示噪声的标准差
    noise = np.random.normal(loc=0, scale=sigma, size=points.shape)

    # （可选）对噪声进行裁剪，防止某些点的扰动过大
    # np.clip 会将数组中的值限制在 [ -clip, clip ] 范围内
    clipped_noise = np.clip(noise, -clip, clip)

    # 将噪声加到原始点云上
    jittered_points = points + clipped_noise

    return jittered_points
class BugTransformer():
    def __init__(self, flame_base_path='/robby/share/Avatar/hongyul/code/GSA/flame_model/assets/flame', device='cuda'):
        self.device = device
        self.flame_model  = FlameHead(
            300,
            100,
            flame_model_path=f'{flame_base_path}/flame2023.pkl',
            flame_lmk_embedding_path=f"{flame_base_path}/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{flame_base_path}/head_template_mesh.obj",
            flame_mask_path=f"{flame_base_path}/FLAME_masks.pkl",

        ).to(self.device)
    def sort_point_cloud(self, xyz, binding):

        # 3. 创建包含原始索引的数组
        indices_with_coords = np.column_stack([xyz, np.arange(len(xyz))])

        # 4. 排序：从下往上(y)，从左往右(x)，从前到后(z)
        # 使用字典序排序：y, x, z (优先级：y > x > z)
        sort_indices = np.lexsort((indices_with_coords[:, 2], indices_with_coords[:, 0], indices_with_coords[:, 1]))

        # 5. 应用排序
        sorted_coords = xyz[sort_indices]
        original_indices = indices_with_coords[sort_indices, 3].astype(int)
        sorted_binding = binding[original_indices]

        # 6. 构建序列：每个点包含 [x, y, z, binding+offset]
        # binding值加上offset避免与坐标token重叠
        # 如果binding是多维的，取第一个值；如果是一维的，直接使用
        if len(sorted_binding.shape) > 1:
            binding_values = sorted_binding[:, 0]  # 取第一列
        else:
            binding_values = sorted_binding
        return sorted_coords, binding_values
    def get_cano_xyz(self, xyz, flame_param, binding):
        flame_param = np.load(flame_param)
        verts, verts_cano = self.flame_model(
            torch.from_numpy(flame_param['shape'])[None, ...].to(self.device),
            torch.zeros(flame_param['expr'][[0]].shape).to(self.device),
            torch.zeros(flame_param['rotation'][[0]].shape).to(self.device),
            torch.zeros(flame_param['neck_pose'][[0]].shape).to(self.device),
            torch.zeros(flame_param['jaw_pose'][[0]].shape).to(self.device),
            torch.zeros(flame_param['eyes_pose'][[0]].shape).to(self.device),
            torch.zeros(flame_param['translation'][[0]].shape).to(self.device),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=None,
            dynamic_offset=None,
        )
        face_center_cano = verts_cano[:, self.flame_model.faces].mean(dim=-2).squeeze(0)
        face_orien_mat_cano, face_scaling_cano = compute_face_orientation(verts_cano.squeeze(0), self.flame_model.faces.squeeze(0), return_scale=True)

        xyz_cano = torch.bmm(face_orien_mat_cano[binding], xyz[..., None]).squeeze(-1)
        xyz_cano = xyz_cano * face_scaling_cano[binding] + face_center_cano[binding]
        xyz_cano  = xyz_cano.detach().cpu().numpy()
        return xyz_cano
    def ply_load(self, path):
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
    def get_load_gs(self, gs_path, flame_para, path):
        xyz, binding = self.ply_load(gs_path)
        xyz_cano = self.get_cano_xyz(torch.tensor(xyz, device=self.device).float(), flame_para, torch.tensor(binding, device=self.device).long().squeeze(-1))
        sorted_xyz, sorted_binding = self.sort_point_cloud(xyz_cano, binding)
        save_ply(path, sorted_xyz, sorted_binding)



# import json
# import os
# def ply_load( path):
#     plydata = PlyData.read(path)
#     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                     np.asarray(plydata.elements[0]["y"]),
#                     np.asarray(plydata.elements[0]["z"])), axis=1)
#     binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
#     if len(binding_names) > 0:
#         binding_names = sorted(binding_names, key=lambda x: int(x.split('_')[-1]))
#         binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
#         for idx, attr_name in enumerate(binding_names):
#             binding[:, idx] = np.asarray(plydata.elements[0][attr_name])
#     return xyz, binding

# if __name__ == '__main__':

#     bug = BugTransformer()
#     file_path_train = "/robby/share/Avatar/hongyul/code/avatarpointillist/configs/cluster_nersamble_2.json"
#     with open(file_path_train, 'r', encoding='utf-8') as f:
#       data = json.load(f)
#     all_id = list(data.keys())  # 获取所有数据的key
#     for i, id in enumerate(all_id):
#       print(id)
#       data_info = data[id]
#       ply_path = os.path.join(data_info['base_path'],'point_cloud_cano_sort.ply')
#       xyz, binding = ply_load(ply_path)
#     #   flame_param_path = os.path.join(data_info['base_path'], 'flame_param.npz')
#     #   save_path = os.path.join(data_info['base_path'],'point_cloud_cano_sort.ply')
#     #   bug.get_load_gs(ply_path, flame_param_path, save_path)



import json
import os
import numpy as np
from plyfile import PlyData

def save_ply(path, xyz, bind):
    l = ['x', 'y', 'z']
    for i in range(1):
        l.append('binding_{}'.format(i))
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes =  xyz

    attributes = np.concatenate((attributes, bind), axis=1)

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def ply_load(path):
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
    else:
        binding = None
    return xyz, binding
import numpy as np

def pad_or_trim_point_cloud_with_binding(xyz, binding, target_num=30000):
    """
    將點雲補齊或裁剪至 target_num 個點，並同步處理對應的 binding。

    Args:
        xyz: (N, 3)
        binding: (N, C)
        target_num: 最終點數

    Returns:
        xyz_new: (target_num, 3)
        binding_new: (target_num, C)
        selected_indices: 被複製或被保留的原始點索引（方便追蹤）
    """
    num_points = xyz.shape[0]

    if num_points == target_num:
        return xyz, binding, np.arange(num_points)

    elif num_points < target_num:
        # 👇 補點邏輯
        num_missing = target_num - num_points
        selected_indices = np.random.choice(num_points, num_missing, replace=True)
        selected_indices.sort()

        xyz_list = []
        binding_list = []
        insert_offset = 0

        for i in range(num_points):
            xyz_list.append(xyz[i])
            binding_list.append(binding[i])

            while insert_offset < num_missing and selected_indices[insert_offset] == i:
                xyz_list.append(xyz[i].copy())
                binding_list.append(binding[i].copy())
                insert_offset += 1

        xyz_new = np.stack(xyz_list, axis=0)
        binding_new = np.stack(binding_list, axis=0)

        return xyz_new, binding_new, np.concatenate([np.arange(num_points), selected_indices])

    else:
        # 👇 裁剪邏輯（隨機選擇 target_num 個點）
        selected_indices = np.random.choice(num_points, target_num, replace=False)
        selected_indices.sort()  # 如需保持順序
        xyz_new = xyz[selected_indices]
        binding_new = binding[selected_indices]
        return xyz_new, binding_new, selected_indices

def pad_point_cloud_with_binding(xyz, binding, target_num=30000):
    """
    补全点云和对应 binding，保持原始顺序结构。

    Args:
        xyz: (N, 3)
        binding: (N, C)
        target_num: 最终目标点数

    Returns:
        xyz_new: (target_num, 3)
        binding_new: (target_num, C)
    """
    num_points = xyz.shape[0]

    if num_points >= target_num:
        return xyz, binding  # 不处理

    num_missing = target_num - num_points

    # 随机选择需要复制的点的索引（有可能重复）
    selected_indices = np.random.choice(num_points, num_missing, replace=True)
    selected_indices.sort()

    xyz_list = []
    binding_list = []
    insert_offset = 0

    for i in range(num_points):
        xyz_list.append(xyz[i])
        binding_list.append(binding[i])

        # 如果当前点被选中复制，就插入副本（可以多个重复）
        while insert_offset < num_missing and selected_indices[insert_offset] == i:
            xyz_list.append(xyz[i].copy())
            binding_list.append(binding[i].copy())
            insert_offset += 1

    xyz_new = np.stack(xyz_list, axis=0)
    binding_new = np.stack(binding_list, axis=0)

    return xyz_new, binding_new, selected_indices

if __name__ == '__main__':

    file_path_train = "/robby/share/Avatar/hongyul/code/avatarpointillist/configs/cluster_nersamble_2.json"
    with open(file_path_train, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_id = list(data.keys())

    point_counts = []

    for i, id in enumerate(all_id):
        print(f"[{i+1}/{len(all_id)}] Processing ID: {id}")
        data_info = data[id]
        ply_path = os.path.join(data_info['base_path'], 'point_cloud_cano_sort.ply')

        if not os.path.exists(ply_path):
            print(f"⚠️ File not found: {ply_path}")
            continue


        xyz_o, binding = ply_load(ply_path)
        xyz, binding, selected_indices = pad_or_trim_point_cloud_with_binding(xyz_o, binding)
        save_path = os.path.join(data_info['base_path'], 'point_cloud_new.npy')
        # save_ply(save_path, xyz, binding)
        np.save(save_path, {
        'xyz': xyz,
        'binding': binding,
        'selected_indices': selected_indices
        })
        num_points = xyz.shape[0]
        point_counts.append(num_points)


    if point_counts:
        max_points = max(point_counts)
        min_points = min(point_counts)
        print(f"\n✅ 点云数量统计完成：")
        print(f"最大点数量：{max_points}")
        print(f"最小点数量：{min_points}")
    else:
        print("❌ 没有成功读取任何点云文件。")
