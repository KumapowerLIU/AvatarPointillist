import os
import numpy as np
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from PIL import Image, ImageFile
import trimesh

import torch
import json
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import NamedTuple, Optional

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None
    flame_param: Optional[dict] = None


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def readCamerasFromTransforms(path, transformsfile, white_background=True,  extension=".png", select_nums=8 ):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if 'camera_angle_x' in contents:
            fovx_shared = contents["camera_angle_x"]
        all_time_index = contents['timestep_indices']
        time_index_select = [0, len(all_time_index)//4, len(all_time_index)//2, len(all_time_index)-1]
        cam_selected = [2, 4, 6, 8, 10, 12]
        time_choice = int(np.random.choice(time_index_select))
        cam_choice = int(np.random.choice(cam_selected))
        input_image_name =f'{str(time_choice).zfill(5)}_{str(cam_choice).zfill(2)}.png'
        sequences = ["EMO-1", "EMO-2", "EMO-3", "EMO-4", "EXP-2", "EXP-3", "EXP-4", "EXP-5", "EXP-8", "EXP-9", "FREE"]

        # Find which of the known sequence folders actually exist under `path`
        existing_sequences = [s for s in sequences if os.path.isdir(os.path.join(path, s))]
        # Randomly choose one existing sequence (use provided seed for reproducibility)
        if len(existing_sequences) > 0:
            selected_sequence = str(np.random.choice(existing_sequences))
        else:
            assert False, f"No known sequence folders found under {path}"
            selected_sequence = None
        input_image_path = os.path.join(path, selected_sequence, "images", input_image_name)

        frames = contents["frames"]

        # If select_nums is provided and less than total frames, randomly sample indices
        total_frames = len(frames)
        if select_nums is not None and 0 < select_nums < total_frames:
            selected_indices = list(np.random.choice(total_frames, size=select_nums, replace=False))
            selected_indices.sort()
        else:
            selected_indices = list(range(total_frames))
        cano_flame_param =  dict(np.load(os.path.join(path, frames[0]['flame_param_path']), allow_pickle=True))

        for list_idx, frame_idx in tqdm(enumerate(selected_indices), total=len(selected_indices)):
            frame = frames[frame_idx]
            file_path = frame["file_path"]
            if extension not in frame["file_path"]:
                file_path += extension
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            if 'w' in frame and 'h' in frame:
                image = None
                width = frame['w']
                height = frame['h']
            else:
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                width, height = image.size

            if 'camera_angle_x' in frame:
                fovx = frame["camera_angle_x"]
            else:
                fovx = fovx_shared
            fovy = focal2fov(fov2focal(fovx, width), height)

            timestep = frame["timestep_index"] if 'timestep_index' in frame else None
            camera_id = frame["camera_index"] if 'camera_id' in frame else None
            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))

            cam_infos.append(CameraInfo(
                uid=frame_idx, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image,
                image_path=image_path, image_name=image_name,
                width=width, height=height,
                timestep=timestep, camera_id=camera_id, flame_param=flame_param))
    return cam_infos, cano_flame_param, input_image_path


def readMeshesFromTransforms(path, transformsfile):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]

        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue

            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos




class ConsistentPointSampler:
    """
    Perform consistent point sampling on a series of meshes with the same topology.

    This class precomputes sampling positions (face indices and barycentric coordinates)
    on a template mesh during initialization. Subsequently, the `sample` method can
    quickly apply these sampling positions to any new mesh with the same topology.
    """
    def __init__(self, template_vertices, faces, face_npy_path,
                index_npy_path , n_points=8192):
        """
        Initialize and precompute the sampling template.

        Args:
            template_vertices (np.ndarray or torch.Tensor): Template mesh vertices, shape [V, 3].
            faces (np.ndarray or torch.Tensor): Mesh face indices, shape [F, 3].
            n_points (int): Number of points to sample.
        """
        print(f"Precomputing consistent sampling template for {n_points} points...")

        if isinstance(template_vertices, torch.Tensor):
            template_vertices = template_vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()

        self.n_points = n_points
        self.faces = torch.from_numpy(faces.astype(np.int64))

        # template_mesh = trimesh.Trimesh(vertices=template_vertices, faces=faces)

        # ==================== Code Correction ====================
        # 1. Use trimesh to randomly sample on template mesh once, getting sample point coords and face indices
        # sampled_points, face_indices = trimesh.sample.sample_surface(template_mesh, self.n_points)
        sampled_points = np.load(index_npy_path)
        face_indices = np.load(face_npy_path)
        print(f"Loaded sampled points from {index_npy_path} and face indices from {face_npy_path}")

        # 2. Get vertices of triangles for sampled points
        triangles = template_vertices[faces[face_indices]] # Shape: [n_points, 3, 3]

        # 3. Use returned sampled_points to compute barycentric coordinates within corresponding triangles
        #    This is a more direct and rigorous approach.
        barycentric_coords = trimesh.triangles.points_to_barycentric(
            triangles=triangles,
            points=sampled_points
        )
        # ======================================================

        # 4. Save face indices and barycentric coordinates as sampling template
        self.face_indices = torch.from_numpy(face_indices.astype(np.int64))
        self.barycentric_coords = torch.from_numpy(barycentric_coords.astype(np.float32))

        print("Sampling template precomputation completed.")

    def to(self, device):
        """Move internal tensors to specified device (e.g., 'cuda:0')"""
        self.faces = self.faces.to(device)
        self.face_indices = self.face_indices.to(device)
        self.barycentric_coords = self.barycentric_coords.to(device)
        return self

    def sample(self, vertices):
        """
        Sample on given new vertices, returning sampled point coordinates and normals.

        Args:
            vertices (torch.Tensor): New mesh vertices, shape [B, V, 3] or [V, 3].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - pc_coor (torch.Tensor): Sampled point coordinates, shape [B, n_points, 3] or [n_points, 3].
                - normals (torch.Tensor): Normals at sampled points, shape [B, n_points, 3] or [n_points, 3].
        """
        is_batched = vertices.dim() == 3
        if not is_batched:
            vertices = vertices.unsqueeze(0)

        batch_size, n_verts, _ = vertices.shape
        device = vertices.device

        if self.faces.device != device:
            self.to(device)

        verts_faces = vertices[:, self.faces]
        v0, v1, v2 = verts_faces[:, :, 0], verts_faces[:, :, 1], verts_faces[:, :, 2]
        face_normals = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), p=2, dim=-1)

        sampled_normals = face_normals[:, self.face_indices]

        sampled_triangles = vertices[:, self.faces[self.face_indices]]

        bary_coords_expanded = self.barycentric_coords.unsqueeze(0).unsqueeze(-1)
        pc_coor = torch.sum(sampled_triangles * bary_coords_expanded, dim=2)

        if not is_batched:
            pc_coor = pc_coor.squeeze(0)
            sampled_normals = sampled_normals.squeeze(0)

        return pc_coor, sampled_normals
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
        return orientation, scale
    return orientation

from plyfile import PlyData, PlyElement

def save_ply(path, xyz):
    l = ['x', 'y', 'z']
    dtype_full = [(attribute, 'f4') for attribute in l]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = xyz
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
