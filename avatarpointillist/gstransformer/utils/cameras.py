# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import torch
import json
import os
import numpy as np

from pathlib import Path
from typing import Tuple, Literal
from scipy.spatial.transform import Rotation as R


"""
R: (N, 3, 3)
T: (N, 3)
E: (N, 4, 4)
vector: (N, 3)
"""


def compose_extrinsic_R_T(R: torch.Tensor, T: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from R and T.
    Batched I/O.
    """
    RT = torch.cat((R, T.unsqueeze(-1)), dim=-1)
    return compose_extrinsic_RT(RT)


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def decompose_extrinsic_R_T(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into R and T.
    Batched I/O.
    """
    RT = decompose_extrinsic_RT(E)
    return RT[:, :, :3], RT[:, :, 3]


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def camera_normalization_objaverse(normed_dist_to_center, poses: torch.Tensor, ret_transform: bool = False):
    assert normed_dist_to_center is not None
    pivotal_pose = compose_extrinsic_RT(poses[:1])
    dist_to_center = pivotal_pose[:, :3, 3].norm(dim=-1, keepdim=True).item() \
        if normed_dist_to_center == 'auto' else normed_dist_to_center

    # compute camera norm (new version)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -dist_to_center],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = compose_extrinsic_RT(poses)
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    poses = decompose_extrinsic_RT(poses)

    if ret_transform:
        return poses, camera_norm_matrix.squeeze(dim=0)
    return poses


def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy


def build_camera_principle(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return torch.cat([
        RT.reshape(-1, 12),
        fx.unsqueeze(-1), fy.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1),
    ], dim=-1)


def build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    I = torch.stack([
        torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
        torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=torch.float32, device=RT.device).repeat(RT.shape[0], 1),
    ], dim=1)
    return torch.cat([
        E.reshape(-1, 16),
        I.reshape(-1, 9),
    ], dim=-1)


def center_looking_at_camera_pose(
    camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None,
    device: torch.device = torch.device('cpu'),
    ):
    """
    camera_position: (M, 3)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4)
    """
    # by default, looking at the origin and world up is pos-z
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
    up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    z_axis = camera_position - look_at
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.cross(up_world, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return extrinsics


def surrounding_views_linspace(n_views: int, radius: float = 2.0, height: float = 0.8, device: torch.device = torch.device('cpu')):
    """
    n_views: number of surrounding views
    radius: camera dist to center
    height: height of the camera
    return: (M, 3, 4)
    """
    assert n_views > 0
    assert radius > 0

    theta = torch.linspace(-torch.pi / 2, 3 * torch.pi / 2, n_views, device=device)
    projected_radius = math.sqrt(radius ** 2 - height ** 2)
    x = torch.cos(theta) * projected_radius
    y = torch.sin(theta) * projected_radius
    z = torch.full((n_views,), height, device=device)

    camera_positions = torch.stack([x, y, z], dim=1)
    extrinsics = center_looking_at_camera_pose(camera_positions, device=device)

    return extrinsics


def create_intrinsics(
    f: float,
    c: float = None, cx: float = None, cy: float = None,
    w: float = 1., h: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    ):
    """
    return: (3, 2)
    """
    fx = fy = f
    if c is not None:
        assert cx is None and cy is None, "c and cx/cy cannot be used together"
        cx = cy = c
    else:
        assert cx is not None and cy is not None, "cx/cy must be provided when c is not provided"
    fx, fy, cx, cy, w, h = fx/w, fy/h, cx/w, cy/h, 1., 1.
    intrinsics = torch.tensor([
        [fx, fy],
        [cx, cy],
        [w, h],
    ], dtype=dtype, device=device)
    return intrinsics


def get_default_source_camera(dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
    # return: (N, D_cam_raw)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -dist_to_center],
        [0, 1, 0, 0],
    ]], dtype=torch.float32, device=device)
    canonical_camera_intrinsics = create_intrinsics(
        f=0.75,
        c=0.5,
        device=device,
    ).unsqueeze(0)
    source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
    return source_camera.repeat(batch_size, 1)


def get_default_render_cameras(n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
    # return: (N, M, D_cam_render)
    render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
    render_camera_intrinsics = create_intrinsics(
        f=0.75,
        c=0.5,
        device=device,
    ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
    render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
    return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)


def projection_from_intrinsics(K: np.ndarray, image_size: Tuple[int], near: float=0.01, far:float=10, flip_y: bool=False, z_sign=-1):
    """
    Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: up, z: in)
    Args:
        K: Intrinsic matrix, (N, 3, 3)
            K = [[
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1],
                ]
            ]
        image_size: (height, width)
    Output:
        proj = [[
                [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                [0.0,    0.0,     z_sign*(far+near) / (far-near), -2*far*near / (far-near)],
                [0.0,    0.0,     z_sign,                     0.0                     ]
            ]
        ]
    """

    B = K.shape[0]
    h, w = image_size

    if K.shape[-2:] == (3, 3):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
    elif K.shape[-1] == 4:
        # fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        fx = K[..., [0]]
        fy = K[..., [1]]
        cx = K[..., [2]]
        cy = K[..., [3]]
    else:
        raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    proj = np.zeros([B, 4, 4])
    proj[:, 0, 0]  = fx * 2 / w
    proj[:, 1, 1]  = fy * 2 / h
    proj[:, 0, 2]  = (w - 2 * cx) / w
    proj[:, 1, 2]  = (h - 2 * cy) / h
    proj[:, 2, 2]  = z_sign * (far+near) / (far-near)
    proj[:, 2, 3]  = -2*far*near / (far-near)
    proj[:, 3, 2]  = z_sign

    if flip_y:
        proj[:, 1, 1] *= -1
    return proj


class OrbitCamera:
    def __init__(
            self,
            W: int,
            H: int,
            r: float = 2,
            fovy: float = 60,
            znear: float = 0.01,
            zfar: float = 10,
            convention: Literal["opengl", "opencv"]="opengl", save_path='camera.json'
        ):
        self.image_width = W
        self.image_height = H
        self.radius_default = r
        self.fovy_default = fovy
        self.znear = znear
        self.zfar = zfar
        self.convention = convention
        self.save_path = save_path

        self.reset()
        self.load()

    def reset(self):
        """ The internal state of the camera is based on the OpenGL convention, but
            properties are converted to the target convention when queried.
        """
        self.rot = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # OpenGL convention
        self.look_at = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.radius = self.radius_default  # camera distance from center
        self.fovy = self.fovy_default
        if self.convention == "opencv":
            self.z_sign = 1
            self.y_sign = 1
        elif self.convention == "opengl":
            self.z_sign = -1
            self.y_sign = -1
        else:
            raise ValueError(f"Unknown convention: {self.convention}")

    def save(self):
        save_dict = {
            'rotation': self.rot.as_matrix().tolist(),
            'look_at': self.look_at.tolist(),
            'radius': self.radius,
            'fovy': self.fovy,
        }
        with open(self.save_path, "w") as f:
            json.dump(save_dict, f, indent=4)

    def clear(self):
        os.remove(self.save_path)

    def load(self):
        if not Path(self.save_path).exists():
            return
        with open(self.save_path, "r") as f:
            load_dict = json.load(f)
        self.rot = R.from_matrix(np.array(load_dict['rotation']))
        self.look_at = np.array(load_dict['look_at'])
        self.radius = load_dict['radius']
        self.fovy = load_dict['fovy']

    @property
    def fovx(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        fovx = 2 * np.arctan(self.image_width / (2 * focal))
        return np.degrees(fovx)

    @property
    def intrinsics(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.image_width // 2, self.image_height // 2])

    @property
    def projection_matrix(self):
        return projection_from_intrinsics(self.intrinsics[None], (self.image_height, self.image_width), self.znear, self.zfar, z_sign=self.z_sign)[0]

    @property
    def world_view_transform(self):
        return np.linalg.inv(self.pose)  # world2cam

    @property
    def full_proj_transform(self):
        return self.projection_matrix @ self.world_view_transform

    @property
    def pose(self):
        # first move camera to (0, 0, radius)
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] += self.radius

        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        pose = rot @ pose

        # translate
        pose[:3, 3] -= self.look_at

        if self.convention == "opencv":
            pose[:, [1, 2]] *= -1
        elif self.convention == "opengl":
            pass
        else:
            raise ValueError(f"Unknown convention: {self.convention}")
        return pose

    def orbit_x(self, angle_x):
        axis_x = self.rot.as_matrix()[:3, 0]
        rotvec_x = axis_x * angle_x
        self.rot = R.from_rotvec(rotvec_x) * self.rot

    def orbit_y(self, angle_y):
        axis_y = self.rot.as_matrix()[:3, 1]
        rotvec_y = axis_y * angle_y
        self.rot = R.from_rotvec(rotvec_y) * self.rot

    def orbit_z(self, angle_z):
        axis_z = self.rot.as_matrix()[:3, 2]
        rotvec_z = axis_z * angle_z
        self.rot = R.from_rotvec(rotvec_z) * self.rot

    def trackball(self, p, q, rot_begin=None):
        axis = np.cross(p, q)
        angle = np.arccos(np.dot(p, q))
        rotvec = axis * angle
        if rot_begin is None:
            self.rot = self.rot * R.from_rotvec(rotvec)
        else:
            self.rot = rot_begin * R.from_rotvec(rotvec)

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx=0, dy=0, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        d = np.array([dx, -dy, dz])  # the y axis is flipped
        self.look_at += 2 * self.rot.as_matrix()[:3, :3] @ d * self.radius / self.image_height * math.tan(np.radians(self.fovy) / 2)
