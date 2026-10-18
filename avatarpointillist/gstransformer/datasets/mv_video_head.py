# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
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

import logging
from collections import defaultdict
import os
from avatarpointillist.argeometry.dataset.data_utils import compute_face_orientation
import glob
from typing import Union
import random
import numpy as np
import torch
import json
from PIL import Image
import cv2
from typing import Optional, Union
from plyfile import PlyData, PlyElement
import copy
from avatarpointillist.gstransformer.datasets.base import BaseDataset
from avatarpointillist.gstransformer.utils.proxy import no_proxy
from avatarpointillist.gstransformer.utils.data_prepare import (
    load_cam_pose, img_center_padding,
    pad_to_square, scale_intrs,
    calc_new_tgt_size_by_aspect,
    load_flame_params,
    center_crop_according_to_mask
)
from avatarpointillist.gstransformer.models.rendering.flame_model.flame_GSA  import FlameHead

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
__all__ = ['MV_VideoHeadDataset']


class MV_VideoHeadDataset(BaseDataset):
    def __init__(self, root_dirs: str, meta_path: Optional[Union[str, list]], gs_meta_path: Optional[Union[str, list]], human_model_path: str,
                 sample_side_views: int,
                 render_image_res_low: int, render_image_res_high: int, render_region_size: int,
                 source_image_res: int,
                 repeat_num=1,
                 crop_range_ratio_hw=[1.0, 1.0],
                 aspect_standard=1.0,  # h/w
                 enlarge_ratio=[0.8, 1.2],
                 multiply=16,
                 debug=False,
                 is_val=False,
                 device='cuda',
                 use_aug = False,
                 use_ar_out=False,
                 **kwargs):
        super().__init__(root_dirs, meta_path)
        with open(gs_meta_path, 'r', encoding='utf-8') as f:
            self.gs_file = json.load(f)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        if not (isinstance(render_region_size, list) or isinstance(render_region_size, tuple)):
            render_region_size = render_region_size, render_region_size  # [H, W]
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        self.use_aug = use_aug
        self.uids = self.uids * repeat_num
        # print(self.uids)
        self.crop_range_ratio_hw = crop_range_ratio_hw
        self.debug = debug
        self.aspect_standard = aspect_standard
        self.use_ar_out = use_ar_out
        assert self.render_image_res_low == self.render_image_res_high
        self.render_image_res = self.render_image_res_low
        self.enlarge_ratio = enlarge_ratio
        print(f"MV_VideoHeadDataset, data_len:{len(self.uids)}, repeat_num:{repeat_num}, debug:{debug}, is_val:{is_val}")
        self.multiply = multiply
        # set data deterministic
        self.is_val = is_val
        # self.flame_model.to(device)
        # self.device = device
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
    def load_rgb_image_with_aug_bg(self, rgb_path, mask_path, bg_color, pad_ratio,
                                   max_tgt_size, aspect_standard, enlarge_ratio,
                                   render_tgt_size, multiply, intr):
        rgb = np.array(Image.open(rgb_path))

        interpolation = cv2.INTER_AREA
        if rgb.shape[0] != 1024 and rgb.shape[0] == rgb.shape[1]:
            rgb = cv2.resize(rgb, (1024, 1024), interpolation=interpolation)
        if pad_ratio > 0:
            rgb = img_center_padding(rgb, pad_ratio)
        rgb = rgb / 255.0

        if mask_path is not None:
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path)) > 180
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                assert pad_ratio == 0
                # if pad_ratio > 0:
                #     mask = img_center_padding(mask, pad_ratio)
                # mask = mask / 255.0
            else:
                # print("no mask file")
                mask = (rgb >= 0.99).sum(axis=2) == 3
                mask = np.logical_not(mask)
                # erode
                mask = (mask * 255).astype(np.uint8)
                kernel_size, iterations = 3, 7
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=iterations) / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]   # [H, W]
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        mask = (mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # if h != w, padding to [a, a] for a = max(h, w)
        rgb, mask, intr = pad_to_square(rgb, mask, bg_color, intr)

        # crop image to enlarge face area.
        try:
            rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
                rgb, mask, aspect_standard, enlarge_ratio
            )
            intr[0, 2] -= offset_x
            intr[1, 2] -= offset_y
        except Exception as ex:
            print(rgb_path, mask_path, ex)

        # resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
            cur_hw=rgb.shape[:2], aspect_standard=aspect_standard,
            tgt_size=render_tgt_size, multiply=multiply
        )
        rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)

        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)

        return rgb, mask, intr

    def uniform_sample_in_chunk(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(np.random.choice(chunk))
        return select_list

    def uniform_sample_in_chunk_det(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(chunk[len(chunk)//2])
        return select_list

    @no_proxy
    def old_inner_get_item(self, idx):
        """
        Args:
            idx: Tuple containing (index, img_per_input)
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        # crop_ratio_h, crop_ratio_w = self.crop_range_ratio_hw

        idx, num_images = idx
        # num_images = 8

        uid = self.uids[idx]
        if len(uid.split('/')) == 1:
            parent_id, child_seq = uid.split('_')
            uid = os.path.join(self.root_dirs, parent_id, child_seq)

            # uid = os.path.join(self.root_dirs, uid)
        mode_str = "train" if not self.is_val else "test"
        # transforms_json = os.path.join(uid, f"transforms_{mode_str}.json")
        transforms_json = os.path.join(uid, "transforms.json")

        with open(transforms_json) as fp:
            data = json.load(fp)
        # cor_flame_path = transforms_json.replace('transforms_{}.json'.format(mode_str),'canonical_flame_param.npz')
        cor_flame_path = transforms_json.replace('transforms.json','canonical_flame_param.npz')
        flame_param = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_param['shape'])
        # data['static_offset'] = flame_param['static_offset']

        all_frames = data["frames"]

        sample_total_views = self.sample_side_views + num_images
        if len(all_frames) >= self.sample_side_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(len(all_frames)))
                else:
                    replace = len(all_frames) < sample_total_views
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(len(all_frames)))
                else:
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = len(all_frames) < sample_total_views
                frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > 1:
                    frame_id_list = np.linspace(0, len(all_frames) - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for i in range(sample_total_views)]

        cam_id_list = frame_id_list
        assert self.sample_side_views + num_images == len(frame_id_list)

        # use teeth or not
        teeth_bs_pth = os.path.join(uid, "tracked_teeth_bs.npz")
        use_teeth = False
        if os.path.exists(teeth_bs_pth) and use_teeth:
            teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
        else:
            teeth_bs_lst = None

        # render images
        c2ws, intrs, rgbs, bg_colors, masks, flame_params = [], [], [], [], [], []

        for frame_id in frame_id_list:
            frame_info = all_frames[frame_id]
            frame_path = os.path.join(uid, frame_info["file_path"])
            if 'nersemble' in frame_path or "tiktok_v34" in frame_path:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])
            else:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])

            teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
            flame_path = os.path.join(uid, frame_info["flame_param_path"])
            flame_param = load_flame_params(flame_path, teeth_bs)

            # if cam_id == 0:
            #     shape_param = flame_param["betas"]

            c2w, ori_intrinsic = load_cam_pose(frame_info, transpose_R="nersemble" in frame_path)

            # bg_color = random.choice([0.0, 0.5, 1.0])  # 1.0
            bg_color = np.random.choice([0.0, 0.5, 1.0])

            # if self.is_val:
            #     bg_color = 1.0
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                frame_path,
                mask_path=mask_path,
                bg_color=bg_color,
                pad_ratio=0,
                max_tgt_size=None,
                aspect_standard=self.aspect_standard,
                enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                render_tgt_size=self.render_image_res,
                multiply=16,
                intr=ori_intrinsic.clone()
            )

            c2ws.append(c2w)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            intrs.append(intrinsic)
            flame_params.append(flame_param)
            masks.append(mask)

        c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
        intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        masks = torch.cat(masks, dim=0)  # [N, 1, H, W]

        flame_params_tmp = defaultdict(list)
        for flame in flame_params:
            for k, v in flame.items():
                flame_params_tmp[k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        flame_params = flame_params_tmp
        # TODO check different betas for same person
        flame_params["betas"] = shape_param


        # reference images
        # frame_id_source_list = random.sample(frame_id_list, num_images)
        frame_id_source_list = np.random.choice(frame_id_list, size=num_images, replace=False).tolist()
        cam_id_source_list = frame_id_source_list

        # prob_refidx = np.ones(self.sample_side_views + 1)
        # if not self.is_val:
        #     prob_refidx[0] = 0.5  # front_prob
        # else:
        #     prob_refidx[0] = 1.0
        # # print(frame_id_list, kinect_color_list, prob_refidx[0])
        # prob_refidx[1:] = (1 - prob_refidx[0]) / len(prob_refidx[1:])
        # ref_idx = np.random.choice(self.sample_side_views + 1, p=prob_refidx)
        # cam_id_source_list = cam_id_list[ref_idx: ref_idx + 1]
        # frame_id_source_list = frame_id_list[ref_idx: ref_idx + 1]

        source_c2ws, source_intrs, source_rgbs, source_flame_params = [], [], [], []
        for cam_id, frame_id in zip(cam_id_source_list, frame_id_source_list):
            frame_info = all_frames[frame_id]
            frame_path = os.path.join(uid, frame_info["file_path"])
            if 'nersemble' in frame_path:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])
            else:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")
            flame_path = os.path.join(uid, frame_info["flame_param_path"])

            teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
            flame_param = load_flame_params(flame_path, teeth_bs)

            c2w, ori_intrinsic = load_cam_pose(frame_info, transpose_R="nersemble" in frame_path)

            # bg_color = 1.0
            # bg_color = 0.0
            # bg_color = random.choice([0.0, 0.5, 1.0])   # 1.
            bg_color = np.random.choice([0.0, 0.5, 1.0])

            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                frame_path,
                mask_path=mask_path,
                bg_color=bg_color,
                pad_ratio=0,
                max_tgt_size=None,
                aspect_standard=self.aspect_standard,
                enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                render_tgt_size=self.source_image_res,
                multiply=self.multiply,
                intr=ori_intrinsic.clone()
            )

            source_c2ws.append(c2w)
            source_intrs.append(intrinsic)
            source_rgbs.append(rgb)
            source_flame_params.append(flame_param)

        source_c2ws = torch.stack(source_c2ws, dim=0)
        source_intrs = torch.stack(source_intrs, dim=0)
        source_rgbs = torch.cat(source_rgbs, dim=0)

        flame_params_tmp = defaultdict(list)
        for flame in source_flame_params:
            for k, v in flame.items():
                flame_params_tmp['source_'+k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        source_flame_params = flame_params_tmp
        # TODO check different betas for same person
        source_flame_params["source_betas"] = shape_param

        render_image = rgbs
        render_mask = masks
        tgt_size = render_image.shape[2:4]   # [H, W]
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 1.1, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 1.1, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        ret = {
            'uid': uid,
            'source_c2ws': source_c2ws,  # [N_i, 4, 4]
            'source_intrs': source_intrs,  # [N_i, 4, 4]
            'source_rgbs': source_rgbs.clamp(0, 1),   # [N_i, 3, H, W]
            'render_image': render_image.clamp(0, 1), # [N, 3, H, W]  (N = N_i + sample_side_views)
            'render_mask': render_mask.clamp(0, 1), #[ N, 1, H, W]
            'c2ws': c2ws,  # [N, 4, 4]
            'intrs': intrs,  # [N, 4, 4]
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + num_images, 1),  # [N, 2]
            'render_bg_colors': bg_colors, # [N, 3]
            'pytorch3d_transpose_R': torch.Tensor(["nersemble" in frame_path]), # [1]


        }

        #['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans', 'betas']
        # 'flame_params': flame_params, # dict: body_pose:[N, 21, 3],
        ret.update(flame_params)
        ret.update(source_flame_params)

        return ret

    # ==========================================================================================================================================

    @no_proxy
    def inner_get_item(self, idx):
        """
        轻量级鲁棒性改进版本
        """
        try:
            return self._inner_get_item_core(idx)
        except Exception as e:
            # 只在最外层捕获所有异常，避免训练中断
            logging.warning(f"Data loading failed for idx {idx}: {str(e)}")
            return self._get_fallback_sample(idx)

    def _inner_get_item_core(self, idx):
        """
        核心逻辑，只在必要位置添加检查
        """
        # idx, num_images = idx
        num_images = 1
        # 1. 基础路径处理（原逻辑保持不变）
        uid = self.uids[idx]
        if len(uid.split('/')) == 1:
            parent_id, child_seq = uid.split('_')
            uid = os.path.join(self.root_dirs, parent_id, child_seq)
            # uid = os.path.join(self.root_dirs, uid)

        # 2. 关键文件存在性检查（只检查最关键的）
        transforms_json = os.path.join(uid, "transforms.json")
        if not os.path.exists(transforms_json):
            raise FileNotFoundError(f"transforms.json missing: {transforms_json}")

        # 3. 加载transforms.json（让JSON解析自然失败）
        with open(transforms_json) as fp:
            data = json.load(fp)

        # 4. 加载canonical flame参数（添加轻量级检查）
        cor_flame_path = transforms_json.replace('transforms.json','canonical_flame_param.npz')
        if os.path.exists(cor_flame_path):
            flame_param = np.load(cor_flame_path)
            shape_param = torch.FloatTensor(flame_param.get('shape', np.zeros(100)))
        else:
            raise FileNotFoundError(f"canonical_flame_param.npz missing: {cor_flame_path}")

            # 使用默认shape参数
            # shape_param = torch.zeros(100, dtype=torch.float32)
            # logging.warning(f"Using default shape param for {uid}")

        # 5. frames验证（只做基本检查）
        all_frames = data["frames"]
        if not all_frames:
            raise ValueError("No frames available")

        # 6. 采样逻辑（保持原有逻辑）
        sample_total_views = self.sample_side_views + num_images
        # frame_id_list = self._sample_frames(all_frames, sample_total_views)
        if len(all_frames) >= self.sample_side_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(len(all_frames)))
                else:
                    replace = len(all_frames) < sample_total_views
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(len(all_frames)))
                else:
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = len(all_frames) < sample_total_views
                frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > 1:
                    frame_id_list = np.linspace(0, len(all_frames) - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for i in range(sample_total_views)]

        cam_id_list = frame_id_list
        assert self.sample_side_views + num_images == len(frame_id_list)

        # 7. teeth参数加载（可选，失败不影响主流程）
        # teeth_bs_lst = self._load_teeth_safely(uid)
        teeth_bs_pth = os.path.join(uid, "tracked_teeth_bs.npz")
        use_teeth = False
        if os.path.exists(teeth_bs_pth) and use_teeth:
            teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
        else:
            teeth_bs_lst = None

        # 8. 批量加载图像数据（在这里处理单个图像失败）
        render_data = self._load_images_batch(
            all_frames, frame_id_list, uid, teeth_bs_lst, shape_param,
            target_size=self.render_image_res
        )

        # 9. 加载源图像
        source_data = self._load_source_images_batch(
            all_frames, frame_id_list, uid, teeth_bs_lst, shape_param,
            num_images, target_size=self.source_image_res
        )
        # print(uid)
        gs_id = f"sample_{parent_id}"
        # gs_id = f"sample_{58:03d}"
        data_info = self.gs_file[gs_id]
        flame_param_path = os.path.join(data_info['base_path'], 'flame_param.npz')

        if self.use_aug:
            index_path = os.path.join(data_info['base_path'],'point_cloud_new.npy')
            ply_path = os.path.join(data_info['base_path'],'point_cloud_cano_sort.ply')
            gs_data =  self._load_gs_npz(index_path, None, None)

        elif self.use_ar_out:
            index_path = os.path.join(data_info['base_path'],'new_point_data.npy')
            ar_feature_path = os.path.join(data_info['base_path'],'topk_1_topp_0.95_temp_0.0_embed.npy')
            lbs_shape_path =  os.path.join(data_info['base_path'],'gs_interp.npz')
            gs_data =  self._load_gs_npz(index_path, lbs_shape_path, ar_feature_path)
        else:
            ply_path = os.path.join(data_info['base_path'],'point_cloud_cano_sort.ply')
            gs_data =  self._load_gs(ply_path, flame_param_path)
        # print( ply_path)

        # print(gs_data['gs_xyz'].shape)
        # print(gs_data['gs_binding'].shape)
        # 10. 组装结果
        return self._assemble_final_result(render_data, source_data, uid, all_frames, gs_data)
    def _load_gs(self, gs_path, flame_para=None):
        xyz, binding = self.ply_load(gs_path)
        binding = binding.squeeze(-1)
        sorted_xyz, sorted_binding = xyz, binding

        # xyz_cano = self.get_cano_xyz(torch.tensor(xyz, device=self.device).float(), flame_para, torch.tensor(binding, device=self.device).long().squeeze(-1))
        # sorted_xyz, sorted_binding = self.sort_point_cloud(xyz_cano , binding)
        # if self.use_aug:
        #     sorted_xyz = jitter_point_cloud(sorted_xyz , sigma=0.01, clip=0.03)
        gs = {
            'gs_xyz': torch.from_numpy(sorted_xyz).float(),  # [N, 3]
            'gs_binding': torch.from_numpy(sorted_binding).long(),  # [N] or [N, k]
        }
        return gs
    def _load_gs_npz(self, index_apth, lbs_shape_path, ar_feature_path, target_num=30000):
        data = np.load(index_apth,   allow_pickle=True).item()
        if not self.use_ar_out:
            selected_indices = data['selected_indices']
            xyz = data['xyz']          # (N, 3)
            binding = data['binding']  # (N, C)
            binding = binding.squeeze(-1)
            sorted_xyz, sorted_binding = xyz, binding
            gs = {
            'gs_xyz': torch.from_numpy(sorted_xyz).float(),  # [N, 3]
            'gs_binding': torch.from_numpy(sorted_binding).long(),  # [N] or [N, k]
            'ar_features': torch.zeros(len(sorted_xyz) * 4, 1024)
            }
        else:
            xyz = data['xyz']
            binding = data['binding']

            binding = binding
            print(binding.shape)
            print(binding.min(), binding.max())

            ar_feaures = np.load( ar_feature_path,  mmap_mode='r' )
            ar_feaures = ar_feaures.squeeze(0)[3:3 + 4 * xyz.shape[0], :]
            seq_len = ar_feaures.shape[0] // 4
            if ar_feaures.shape[0] != 4 * xyz.shape[0]:
                raise ValueError("AR features must contain exactly four predictor states per point")
            sorted_xyz, sorted_binding = xyz, binding

            data_lbs_shape = np.load(lbs_shape_path,  mmap_mode='r')
            my_lbs_weights = torch.from_numpy(data_lbs_shape['my_lbs_weights']).float()
            my_shapedirs_up = torch.from_numpy(data_lbs_shape['my_shapedirs_up']).float()
            gs = {
            'gs_xyz': torch.from_numpy(sorted_xyz).float(),  # [N, 3]
            'gs_binding': torch.from_numpy(sorted_binding).long(),  # [N] or [N, k]
            'ar_features': torch.from_numpy(ar_feaures).float(),  # [4N, C]
            'my_lbs_weights': my_lbs_weights,  # [N, 104]
            'my_shapedirs_up': my_shapedirs_up,  # [N,
            }


        return gs
    def _load_images_batch(self, all_frames, frame_id_list, uid, teeth_bs_lst, shape_param, target_size):
        """批量加载图像，只在单个图像失败时重试"""
        c2ws, intrs, rgbs, bg_colors, masks, flame_params = [], [], [], [], [], []

        failed_indices = []

        for i, frame_id in enumerate(frame_id_list):
            try:
                frame_info = all_frames[frame_id]
                frame_path = os.path.join(uid, frame_info["file_path"])

                if 'nersemble' in frame_path or "tiktok_v34" in frame_path:
                    mask_path = os.path.join(uid, frame_info["fg_mask_path"])
                else:
                    mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(uid, frame_info["fg_mask_path"])

                teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
                flame_path = os.path.join(uid, frame_info["flame_param_path"])
                flame_param = load_flame_params(flame_path, teeth_bs)

                c2w, ori_intrinsic = load_cam_pose(frame_info, transpose_R="nersemble" in frame_path)
                bg_color = np.random.choice([0.0, 0.5, 1.0])

                rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                    frame_path, mask_path=mask_path, bg_color=bg_color,
                    pad_ratio=0, max_tgt_size=None, aspect_standard=self.aspect_standard,
                    enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                    render_tgt_size=target_size, multiply=16, intr=ori_intrinsic.clone()
                )

                c2ws.append(c2w)
                intrs.append(intrinsic)
                rgbs.append(rgb)
                bg_colors.append(bg_color)
                masks.append(mask)
                flame_params.append(flame_param)

            except Exception as e:
                # 记录失败的索引，稍后替换
                failed_indices.append(i)
                logging.warning(f"Failed to load frame {frame_id}: {str(e)}")

        # 处理失败的图像：用成功加载的图像随机替换
        if failed_indices and len(c2ws) > 0:
            for failed_idx in failed_indices:
                # 随机选择一个成功的图像进行复制
                success_idx = np.random.randint(0, len(c2ws))
                c2ws.insert(failed_idx, c2ws[success_idx].clone())
                intrs.insert(failed_idx, intrs[success_idx].clone())
                rgbs.insert(failed_idx, rgbs[success_idx].clone())
                bg_colors.insert(failed_idx, bg_colors[success_idx])
                masks.insert(failed_idx, masks[success_idx].clone())
                flame_params.insert(failed_idx, flame_params[success_idx].copy())

        # 如果全部失败，抛出异常让上层处理
        if not c2ws:
            raise RuntimeError("All frames failed to load")

        flame_params_tmp = defaultdict(list)
        for flame in flame_params:
            for k, v in flame.items():
                flame_params_tmp[k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        flame_params = flame_params_tmp
        # TODO check different betas for same person
        flame_params["betas"] = shape_param

        return {
            'c2ws': torch.stack(c2ws, dim=0),
            'intrs': torch.stack(intrs, dim=0),
            'rgbs': torch.cat(rgbs, dim=0),
            'bg_colors': torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3),
            'masks': torch.cat(masks, dim=0),
            'flame_params': flame_params,
        }

    def _load_source_images_batch(self, all_frames, frame_id_list, uid, teeth_bs_lst, shape_param, num_images, target_size):
        """加载源图像数据"""
        # 从frame_id_list中随机选择num_images个作为源图像
        # frame_id_source_list = np.random.choice(frame_id_list, size=num_images, replace=False).tolist()

        # directly get the front num_images frames from random/ordered index of frame_id_list
        frame_id_source_list = frame_id_list[:num_images]
        cam_id_source_list = frame_id_source_list

        source_c2ws, source_intrs, source_rgbs, source_flame_params = [], [], [], []
        failed_indices = []

        for i, (cam_id, frame_id) in enumerate(zip(cam_id_source_list, frame_id_source_list)):
            try:
                frame_info = all_frames[frame_id]
                frame_path = os.path.join(uid, frame_info["file_path"])

                if 'nersemble' in frame_path:
                    mask_path = os.path.join(uid, frame_info["fg_mask_path"])
                else:
                    mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")

                teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
                flame_path = os.path.join(uid, frame_info["flame_param_path"])
                flame_param = load_flame_params(flame_path, teeth_bs)

                c2w, ori_intrinsic = load_cam_pose(frame_info, transpose_R="nersemble" in frame_path)
                bg_color = np.random.choice([0.0, 0.5, 1.0])
                rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                    frame_path, mask_path=mask_path, bg_color=bg_color,
                    pad_ratio=0, max_tgt_size=None, aspect_standard=self.aspect_standard,
                    enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                    render_tgt_size=target_size, multiply=self.multiply, intr=ori_intrinsic.clone()
                )
                source_c2ws.append(c2w)
                source_intrs.append(intrinsic)
                source_rgbs.append(rgb)
                source_flame_params.append(flame_param)

            except Exception as e:
                failed_indices.append(i)
                logging.warning(f"Failed to load source frame {frame_id}: {str(e)}")

        # 处理失败的源图像
        if failed_indices and len(source_c2ws) > 0:
            for failed_idx in failed_indices:
                success_idx = np.random.randint(0, len(source_c2ws))
                source_c2ws.insert(failed_idx, source_c2ws[success_idx].clone())
                source_intrs.insert(failed_idx, source_intrs[success_idx].clone())
                source_rgbs.insert(failed_idx, source_rgbs[success_idx].clone())
                source_flame_params.insert(failed_idx, source_flame_params[success_idx].copy())

        if not source_c2ws:
            raise RuntimeError("All source frames failed to load")

        flame_params_tmp = defaultdict(list)
        for flame in source_flame_params:
            for k, v in flame.items():
                flame_params_tmp['source_'+k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        source_flame_params = flame_params_tmp
        # TODO check different betas for same person
        source_flame_params["source_betas"] = shape_param

        return {
            'source_c2ws': torch.stack(source_c2ws, dim=0),
            'source_intrs': torch.stack(source_intrs, dim=0),
            'source_rgbs': torch.cat(source_rgbs, dim=0),
            'source_flame_params': source_flame_params,
            'is_nersemble': torch.Tensor(["nersemble" in frame_path]), # [1]
        }

    def _assemble_final_result(self, render_data, source_data, uid, all_frames, gs_data):
        """组装最终结果"""
        # 获取图像尺寸信息
        render_image = render_data['rgbs']
        render_mask = render_data['masks']
        tgt_size = render_image.shape[2:4]   # [H, W]

        # 基本的断言检查
        intrs = render_data['intrs']
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 1.1, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 1.1, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        # 组装返回结果
        ret = {
            'uid': uid,
            'source_c2ws': source_data['source_c2ws'],  # [N_i, 4, 4]
            'source_intrs': source_data['source_intrs'],  # [N_i, 4, 4]
            'source_rgbs': source_data['source_rgbs'].clamp(0, 1),   # [N_i, 3, H, W]
            'render_image': render_image.clamp(0, 1), # [N, 3, H, W]  (N = N_i + sample_side_views)
            'render_mask': render_mask.clamp(0, 1), #[ N, 1, H, W]
            'c2ws': render_data['c2ws'],  # [N, 4, 4]
            'intrs': render_data['intrs'],  # [N, 4, 4]
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + len(source_data['source_c2ws']), 1),  # [N, 2]
            'render_bg_colors': render_data['bg_colors'], # [N, 3]
            'pytorch3d_transpose_R': source_data['is_nersemble'], # [1]
            'gs_xyz': gs_data['gs_xyz'],   # [N, 3],
            'gs_binding': gs_data['gs_binding'] ,  # [N] or [N, k],
            'ar_features': gs_data['ar_features'],  # [N, k]
            'my_lbs_weights': gs_data['my_lbs_weights'],  # [N, 104]
            'my_shapedirs_up': gs_data['my_shapedirs_up']


        }

        # 添加flame参数
        ret.update(render_data['flame_params'])
        ret.update(source_data['source_flame_params'])

        return ret

    def _get_fallback_sample(self, idx):
        """获取备用样本，避免训练中断"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # 随机选择另一个样本
                fallback_idx = np.random.randint(0, len(self.uids))
                if fallback_idx == idx[0]:
                    fallback_idx = (fallback_idx + 1) % len(self.uids)
                return self._inner_get_item_core((fallback_idx, idx[1]))
            except Exception as e:
                raise RuntimeError("Fallback Sample Fails 5 times!")

    # ==========================================================================================================================================
