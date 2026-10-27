import os
import numpy as np
import tqdm
import trimesh
from torch import nn
import torch
from plyfile import PlyData, PlyElement
import numpy
from PIL import Image
from transformers import AutoImageProcessor
from avatarpointillist.argeometry.dataset.data_utils import ConsistentPointSampler, compute_face_orientation,save_ply
from avatarpointillist.flame_condition.flame import FlameHeadSubdivided, FlameHead
import json
def to_rgb_image(maybe_rgba: Image.Image, target_size: int = 512, pad_size: int = 802):
    """
    Convert input image to RGB and scale to target_size x target_size while maintaining aspect ratio.
    If image has alpha channel, composite on pure white background first; then pad to square (prefer pad_size x pad_size), finally resize to target_size x target_size.
    """
    # Ensure RGBA to handle alpha (convert other modes too)
    if maybe_rgba.mode not in ('RGB', 'RGBA'):
        img_rgba = maybe_rgba.convert('RGBA')
    else:
        img_rgba = maybe_rgba.convert('RGBA')

    # Composite on pure white background (handle transparent channel)
    background = Image.new('RGBA', img_rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(background, img_rgba).convert('RGB')

    w, h = composited.size

    # Choose square edge length: prefer pad_size when height is pad_size and width < pad_size, otherwise max(w,h)
    if h == pad_size and w < pad_size:
        square = pad_size
    else:
        square = max(w, h)

    if square != w or square != h:
        canvas = Image.new('RGB', (square, square), (255, 255, 255))
        left = (square - w) // 2
        top = (square - h) // 2
        canvas.paste(composited, (left, top))
        composited = canvas

    # Resize to target size
    try:
        resample = Image.LANCZOS
    except AttributeError:
        resample = Image.ANTIALIAS
    resized = composited.resize((target_size, target_size), resample=resample)
    return resized

# def to_rgb_image(maybe_rgba: Image.Image):
#     if maybe_rgba.mode == 'RGB':
#         return maybe_rgba
#     elif maybe_rgba.mode == 'RGBA':
#         rgba = maybe_rgba
#         img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
#         img = Image.fromarray(img, 'RGB')
#         img.paste(rgba, mask=rgba.getchannel('A'))
#         return img
#     else:
#         raise ValueError("Unsupported image type.", maybe_rgba.mode)


def normalize(arr):
    # Calculate L2 norm of each vector
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero, replace zero norm with 1
    norms[norms == 0] = 1
    # Normalize each vector
    return arr / norms


def quantize_coords_torch(coords, num_bins=1024, coord_range=(0.0, 1.0)):
    """
    Quantize continuous coordinates to integer tokens (bin index)
    x-axis: [0, num_bins-1]
    y-axis: [num_bins, 2*num_bins-1] 
    z-axis: [2*num_bins, 3*num_bins-1]

    Args:
        coords: torch.Tensor of shape (N, 3), float32
        num_bins: int, quantization levels per axis (default 1024)
        coord_range: tuple, min and max values of coordinates (default range [0.0, 1.0])

    Returns:
        quantized: torch.Tensor of shape (N, 3), int64, each axis mapped to different bin ranges
    """
    min_val, max_val = coord_range
    coords_clipped = torch.clamp(coords, min=min_val, max=max_val)
    scaled = (coords_clipped - min_val) / (max_val - min_val)
    base_quantized = torch.floor(scaled * (num_bins - 1)).long()
    
    # Add offset to each axis, mapping to different bin ranges
    quantized = torch.zeros_like(base_quantized)
    quantized[:, 0] = base_quantized[:, 0]                    # x-axis: [0, num_bins-1]
    quantized[:, 1] = base_quantized[:, 1] + num_bins         # y-axis: [num_bins, 2*num_bins-1]
    quantized[:, 2] = base_quantized[:, 2] + 2 * num_bins     # z-axis: [2*num_bins, 3*num_bins-1]
    
    return quantized


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


def rotate_mesh_with_normal(vertices):
    angle_y = np.random.uniform(0, 2 * np.pi)

    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])

    rotated_vertices = np.dot(vertices, R_y.T)
    # rotated_pc = np.dot(pc_coor, R_y.T)
    # rotated_normal = np.dot(normal, R_y.T)

    return rotated_vertices  # , rotated_pc, rotated_normal


class DataHeadARGPTTest(nn.Module):
    def __init__(self, args, device, test=False):
        super().__init__()

        self.xyzface_num = 4
        self.binding_offset = args.num_faces
        self.bound_upper = np.array(args.global_max, dtype=np.float32)
        self.bound_lower = np.array(args.global_min, dtype=np.float32)
 
        self.dino_img_processor = AutoImageProcessor.from_pretrained(args.dino_path)
        # Add spatial bounds and quantization parameters
        self.coord_bins = args.n_discrete_size if hasattr(args, 'n_discrete_size') else 512
        self.bos_token_id = self.binding_offset + self.coord_bins
        self.eos_token_id = self.binding_offset + self.coord_bins + 1
        self.pad_token_id = self.binding_offset + self.coord_bins + 2
        self.test = test
        # # Use CUDA if available, otherwise use CPU
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")
        
        self.flame_model_subdivided = FlameHeadSubdivided(
            300,
            100,
            flame_model_path=f'{args.flame_base_path}/flame2023.pkl',
            flame_lmk_embedding_path=f"{args.flame_base_path}/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{args.flame_base_path}/head_template_mesh.obj",
            flame_mask_path=f"{args.flame_base_path}/FLAME_masks.pkl",
            subdivide_num=1
        )
        self.flame_model = FlameHead(
            300,
            100,
            flame_model_path=f'{args.flame_base_path}/flame2023.pkl',
            flame_lmk_embedding_path=f"{args.flame_base_path}/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{args.flame_base_path}/head_template_mesh.obj",
            flame_mask_path=f"{args.flame_base_path}/FLAME_masks.pkl",

        ) 
        
        with open(args.file_path_train, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        # print(self.data)
        # self.data = self.data[0]
        self.all_id = list(self.data.keys())  # 获取所有数据的key

        self.sampler = ConsistentPointSampler(
                template_vertices=self.flame_model_subdivided.v_template_up,
                faces=self.flame_model_subdivided.faces_up,
                n_points=args.n_sample_points,
                face_npy_path=args.face_npy_path,
                index_npy_path=args.index_npy_path,
            )
        self.sampler.to(device)
        self.flame_model.to(device)
        self.flame_model_subdivided.to(device)
        self.device = device
        # all tokens we need the face id to get the target gs

    def sort_point_cloud(self, xyz, binding):
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
        xyz_norm = (xyz - self.bound_lower) / (self.bound_upper - self.bound_lower)
        # xyz_norm = xyz
        # xyz_norm = (xyz - self.spatial_bounds[0]) / (self.spatial_bounds[1] - self.spatial_bounds[0]) 
        # 2. 量化坐标到整数范围 [0, coord_bins-1]
        xyz_quantized = quantize_coords_numpy(xyz_norm, self.coord_bins, (0.0, 1.0))
        
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
        binding_with_offset = binding_values + self.coord_bins
        
        # 7. 创建序列，每个点4个token
        sequence = []
        for i in range(len(sorted_coords)):
            point_tokens = [sorted_coords[i, 0], sorted_coords[i, 1], sorted_coords[i, 2], binding_with_offset[i]]
            sequence.extend(point_tokens)

        
        return np.array(sequence) 

    def __len__(self):

        return len(self.all_id)

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
    @torch.no_grad()
    def get_pc_condition(self, flame_params):
        flame_param = np.load(flame_params)
        shape_params_single = torch.from_numpy(flame_param['shape'])[None, ...].to(self.device)
        verts_CANO_single = self.flame_model_subdivided.get_cano_verts(shape_params_single)
        pc_coor_single, normals_single = self.sampler.sample(verts_CANO_single)
        pc_coor_single = pc_coor_single.squeeze(0).cpu().numpy()
        normals_single = normals_single.squeeze(0).cpu().numpy()
        return pc_coor_single, normals_single, verts_CANO_single
    


    def get_input_condition(self, path):
        # with open(os.path.join(path, 'transforms_train.json')) as json_file:
        #     contents = json.load(json_file)
        #     all_time_index = contents['timestep_indices']
        # time_index_select = [0, len(all_time_index)//4, len(all_time_index)//2, len(all_time_index)-1]
 

        cam_selected = [2, 4, 6, 8, 10, 12]
        # time_choice = int(np.random.choice(time_index_select))
        cam_choice = int(np.random.choice(cam_selected))

        with open(os.path.join(path, 'sequences_trainval.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sequences = [line.strip() for line in lines]
        # sequences = ["EMO-1", "EMO-2", "EMO-3", "EMO-4", "EXP-2", "EXP-3", "EXP-4", "EXP-5", "EXP-8", "EXP-9", "FREE"]

        # Find which of the known sequence folders actually exist under `path`
        parent_path = os.path.dirname(path)
        existing_sequences = [s for s in sequences if os.path.isdir(os.path.join(parent_path, s))]
        # Randomly choose one existing sequence (use provided seed for reproducibility)
        if len(existing_sequences) > 0:
            selected_sequence = str(np.random.choice(existing_sequences))
        else:
            assert False, f"No known sequence folders found under {path}"
        input_image_path_base = os.path.join(parent_path, selected_sequence, "images")
        all_image_nums =  len([f for f in os.listdir(input_image_path_base) if f.endswith('.png')])
        time_index_max = all_image_nums // 16 
        time_index_select = [0, time_index_max//4,time_index_max//2, time_index_max-1]
        time_choice = int(np.random.choice(time_index_select))
        input_image_name =f'{str(time_choice).zfill(5)}_{str(cam_choice).zfill(2)}.png'
        input_image_path = os.path.join(input_image_path_base, input_image_name)
        if self.test:
            print(input_image_path)
        return input_image_path
    @torch.no_grad()
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
 

    def xyz_augment(self, xyz):
        pass

    def __getitem__(self, idx):
        # Get data ID
        data_id = self.all_id[idx]
        data_info = self.data[data_id]  # Get data info dict for this ID
        flame_param_path = data_info['flame_path']
        ply_path =  data_info['ply_path']
        # Load ply file
        xyz, binding = self.ply_load(ply_path)
        xyz_cano = self.get_cano_xyz(torch.tensor(xyz, device=self.device).float(), flame_param_path, torch.tensor(binding, device=self.device).long().squeeze(-1))
        sequence = self.sort_point_cloud(xyz_cano, binding)   
        # Load image file
        face_raw_image = to_rgb_image(Image.open(data_info['img_path']))
        dino_img = self.dino_img_processor(images=face_raw_image, return_tensors="pt").pixel_values
        dino_img = dino_img.squeeze(0)
        # Use new sorting method to process point cloud, now includes feature extraction
 
        # Get point cloud inputs
        pc_coor, pc_normal, flame_verts = self.get_pc_condition(flame_param_path)
        # save_ply('./test_sample.ply', xyz=pc_coor)
        # Add start and end tokens (4 tokens per point, so need 4 start and 4 end tokens)
        # sequence = np.array([self.bos_token_id] * 4)
        sequence = np.concatenate([
            np.array([self.bos_token_id] * 4),  # 4 start tokens
            sequence,                            # sorted point cloud sequence
            np.array([self.eos_token_id] * 4)   # 4 end tokens
        ])
 
 
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.99
        data_out = {
            'dino_img': dino_img,
            'sequence': sequence,  # sorted sequence
            'pc_normal': np.concatenate([pc_coor, pc_normal], axis=-1)

        }
        return data_out
