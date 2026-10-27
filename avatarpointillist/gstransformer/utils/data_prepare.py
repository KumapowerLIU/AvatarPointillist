from collections import defaultdict
import os
import json
import numpy as np
from PIL import Image
import cv2
import torch
import math
from tqdm import tqdm
from typing import Tuple, Optional, List
from decord import VideoReader

from .interpolate_camera import interpolate_multiview_cameras, select_multiview_camera_path
from .matting import SoftForegroundSegmenter


AVATAR_MATTING_MODEL = '/nas5/liuhongyu/AR_Results/ners_test_data/foreground-segmentation-model-vitl16_384.onnx'
foreground_segmenter = SoftForegroundSegmenter(onnx_model=AVATAR_MATTING_MODEL)


def scale_intrs(intrs, ratio_x, ratio_y):
    if len(intrs.shape) >= 3:
        intrs[:, 0] = intrs[:, 0] * ratio_x
        intrs[:, 1] = intrs[:, 1] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y  
    return intrs    


def calc_new_tgt_size(cur_hw, tgt_size, multiply):
    ratio = tgt_size / min(cur_hw)
    tgt_size = int(ratio * cur_hw[0]), int(ratio * cur_hw[1])
    tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def calc_new_tgt_size_by_aspect(cur_hw, aspect_standard, tgt_size, multiply):
    assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
    tgt_size = tgt_size * aspect_standard, tgt_size
    # tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
    tgt_size = round(tgt_size[0] / multiply) * multiply, round(tgt_size[1] / multiply) * multiply
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def img_center_padding(img_np, pad_ratio):
    
    ori_w, ori_h = img_np.shape[:2]
    
    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)
    
    if len(img_np.shape) > 2:
        img_pad_np = np.zeros((w, h, img_np.shape[2]), dtype=np.uint8)
    else:
        img_pad_np = np.zeros((w, h), dtype=np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[offset_h: offset_h + img_np.shape[0]:, offset_w: offset_w + img_np.shape[1]] = img_np
    
    return img_pad_np


def resize_image_keepaspect_np(img, max_tgt_size):
    """
    similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio  
    """
    h, w = img.shape[:2]
    ratio = max_tgt_size / max(h, w)
    new_h, new_w = round(h * ratio), round(w * ratio)
    return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop_according_to_mask(img, mask, aspect_standard, enlarge_ratio):
    """ 
        img: [H, W, 3]
        mask: [H, W]
    """ 
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise Exception("empty mask")

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    
    center_x, center_y = img.shape[1]//2, img.shape[0]//2
    
    half_w = max(abs(center_x - x_min), abs(center_x -  x_max))
    half_h = max(abs(center_y - y_min), abs(center_y -  y_max))
    half_w_raw = half_w
    half_h_raw = half_h
    
    aspect = half_h / half_w
    if aspect >= aspect_standard:                
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    if half_h > center_y:
        half_w = round(half_h_raw / aspect_standard)
        half_h = half_h_raw
    if half_w > center_x:
        half_h = round(half_w_raw * aspect_standard)
        half_w = half_w_raw

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) >  0.01:
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
        enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
        enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
        enlarge_ratio_cur = np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min) + enlarge_ratio_min
        half_h, half_w = round(enlarge_ratio_cur * half_h), round(enlarge_ratio_cur * half_w)

    assert half_h <= center_y
    assert half_w <= center_x
    assert abs(half_h / half_w - aspect_standard) < 0.03
    
    offset_x = center_x - half_w
    offset_y = center_y - half_h
    
    new_img = img[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
    new_mask = mask[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
    
    return  new_img, new_mask, offset_x, offset_y


def preprocess_image(rgb_path, mask_path, intr, pad_ratio, bg_color,
                     max_tgt_size, aspect_standard, enlarge_ratio,
                     render_tgt_size, multiply, need_mask=True,
                     get_shape_param=False):
    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()
    if pad_ratio > 0:
        rgb = img_center_padding(rgb, pad_ratio)

    rgb = rgb / 255.0
    if need_mask:
        if rgb.shape[2] < 4:
            if mask_path is not None:
                # mask = np.array(Image.open(mask_path))
                mask = np.array(Image.open(mask_path)) > 180
            else:
                # from rembg import remove
                # mask = remove(rgb_raw[:, :, (2, 1, 0)])[:, :, -1]  # np require [bgr]
                # print("rmbg mask: ", mask.min(), mask.max(), mask.shape)
                # foreground_segmenter = SoftForegroundSegmenter(onnx_model=AVATAR_MATTING_MODEL)
                mask = foreground_segmenter.estimate_foreground_segmentation(cv2.imread(rgb_path))
                cv2.imwrite('./mask.png', mask * 255)
            if pad_ratio > 0:
                mask = img_center_padding(mask, pad_ratio)
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]   # [H, W]
    else:
        # just placeholder
        mask = np.ones_like(rgb[:, :, 0])
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]

    # mask = (mask > 0.5).astype(np.float32)
    mask = mask.astype(np.float32)
    if (rgb.shape[0] == rgb.shape[1]) and (rgb.shape[0]==512):
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # # resize to specific size require by preprocessor of flame-estimator.
    # rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    # mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # if h != w, padding to [a, a] for a = max(h, w)
    rgb, mask, intr = pad_to_square(rgb, mask, bg_color, intr)

    # crop image to enlarge face area.
    try:
        rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
            rgb, mask, aspect_standard, enlarge_ratio
        )
        if intr is not None:
            intr[0, 2] -= offset_x
            intr[1, 2] -= offset_y
    except Exception as ex:
        print(rgb_path, mask_path, ex)

    # resize to render_tgt_size for training
    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2], aspect_standard=aspect_standard,
        tgt_size=render_tgt_size, multiply=multiply
    )
    rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)
    
    if intr is not None:
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2
    
    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

    # read shape_param
    shape_param = None
    if get_shape_param:
        cor_flame_path = os.path.join(os.path.dirname(os.path.dirname(rgb_path)), 'canonical_flame_param.npz')
        flame_p = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_p['shape'])

    return rgb, mask, intr, shape_param


def extract_imgs_from_video(video_file, save_root, fps):
    print(f"extract_imgs_from_video:{video_file}")
    vr = VideoReader(video_file)
    for i in range(0, len(vr), fps):
        frame = vr[i].asnumpy()
        save_path = os.path.join(save_root, f"{i:05d}.jpg")
        cv2.imwrite(save_path, frame[:, :, (2, 1, 0)])


def predict_motion_seqs_from_images(image_folder:str, save_root, fps=6):
    id_name = os.path.splitext(os.path.basename(image_folder))[0]
    if os.path.isfile(image_folder) and (image_folder.endswith("mp4") or image_folder.endswith("move")):
        save_frame_root = os.path.join(save_root, "extracted_frames", id_name)
        if not os.path.exists(save_frame_root):
            os.makedirs(save_frame_root, exist_ok=True)
            extract_imgs_from_video(video_file=image_folder, save_root=save_frame_root, fps=fps)
        else:
            print("skip extract_imgs_from_video......")
        image_folder = save_frame_root

    image_folder_abspath = os.path.abspath(image_folder)
    print(f"predict motion seq:{image_folder_abspath}")
    save_flame_root = image_folder + "_flame_params_mhmr"
    if not os.path.exists(save_flame_root):
        cmd = f"cd thirdparty/multi-hmr &&  python infer_batch.py  --data_root {image_folder_abspath}  --out_folder {image_folder_abspath} --crop_head   --crop_hand   --pad_ratio 0.2 --smplify"
        os.system(cmd)
    else:
        print("skip predict flame.........")
    return save_flame_root, image_folder


def render_flame_mesh(data, render_intrs, c2ws,
                      is_nersemble_cam: bool = False,
                      human_model_path: str = "./model_zoo/human_parametric_models"):
    from gstransformer.models.rendering.flame_model.flame import FlameHead, FlameHeadSubdivided
    from gstransformer.models.rendering.utils.vis_utils import render_mesh

    subdivide = 0  # 2
    flame_sub_model = FlameHeadSubdivided(
        300,
        100,
        add_teeth=True,
        add_shoulder=False,
        flame_model_path='/nas5/liuhongyu/AR_Results/ners_test_data/ar_out/flame/flame_assets/flame/flame2023.pkl',
        flame_lmk_embedding_path="/nas5/liuhongyu/AR_Results/ners_test_data/ar_out/flame/flame_assets/flame/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path="/nas5/liuhongyu/AR_Results/ners_test_data/ar_out/flame/flame_assets/flame/head_template_mesh.obj",
        flame_parts_path="/nas5/liuhongyu/AR_Results/ners_test_data/ar_out/flame/flame_assets/flame/FLAME_masks.pkl",
        subdivide_num=subdivide
    ).cuda()

    shape = data['betas'].to('cuda')
    flame_param = {}
    flame_param['expr'] = data['expr'].to('cuda')
    flame_param['rotation'] = data['rotation'].to('cuda')
    flame_param['neck'] = data['neck_pose'].to('cuda')
    flame_param['jaw'] = data['jaw_pose'].to('cuda')
    flame_param['eyes'] = data['eyes_pose'].to('cuda')
    flame_param['translation'] = data['translation'].to('cuda')
    
    v_cano = flame_sub_model.get_cano_verts(
        shape.unsqueeze(0)
    )

    ret = flame_sub_model.animation_forward(
        v_cano.repeat(flame_param['expr'].shape[0], 1, 1),
        shape.unsqueeze(0).repeat(flame_param['expr'].shape[0], 1),
        flame_param['expr'],
        flame_param['rotation'],
        flame_param['neck'],
        flame_param['jaw'],
        flame_param['eyes'],
        flame_param['translation'],
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        # static_offset=batch_data['static_offset'].to('cuda'),
        static_offset=None,
    )

    flame_face = flame_sub_model.faces.cpu().squeeze().numpy()
    mesh_render_list = []
    num_view = flame_param['expr'].shape[0]
    print("rendering motion mesh...")
    for v_idx in tqdm(range(num_view)):
        intr = render_intrs[v_idx]
        cam_param = {"focal": torch.tensor([intr[0, 0], intr[1, 1]]), 
                    "princpt": torch.tensor([intr[0, 2], intr[1, 2]])}
        render_shape = int(cam_param['princpt'][1]* 2), int(cam_param['princpt'][0] * 2)     # require h, w

        vertices = ret["animated"][v_idx].cpu().squeeze()

        c2w = c2ws[v_idx]
        w2c = torch.inverse(c2w)
        R = w2c[:3, :3]
        R = R.transpose(1, 0) if is_nersemble_cam else R
        T = w2c[:3, 3]
        vertices = vertices @ R + T

        mesh_render, is_bkg = render_mesh(
            vertices, flame_face, cam_param,
            np.ones((render_shape[0],render_shape[1], 3), dtype=np.float32)*255,
            return_bg_mask=True)
        
        mesh_render = mesh_render.astype(np.uint8)
        mesh_render_list.append(mesh_render)
    mesh_render = np.stack(mesh_render_list)
    print("render motion mesh finished")
    return mesh_render


def load_cam_pose(frame_info, transpose_R=False):
    c2w = torch.eye(4)
    c2w = np.array(frame_info["transform_matrix"])
    c2w[:3, 1:3] *= -1
    c2w = torch.FloatTensor(c2w)
    
    # TODO: CHECK HERE! 
    # if transpose_R:
    #     w2c = torch.inverse(c2w)
    #     w2c[:3, :3] = w2c[:3, :3].transpose(1, 0).contiguous()
    #     c2w = torch.inverse(w2c)
    
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = frame_info["fl_x"]
    intrinsic[1, 1] = frame_info["fl_y"]
    intrinsic[0, 2] = frame_info["cx"]
    intrinsic[1, 2] = frame_info["cy"]
    intrinsic = intrinsic.float()
    
    return c2w, intrinsic


def load_flame_params(flame_file_path, teeth_bs=None):
    flame_param = dict(np.load(flame_file_path, allow_pickle=True))
    flame_param_tensor = {}
    flame_param_tensor['expr'] = torch.FloatTensor(flame_param['expr'])[0]
    flame_param_tensor['rotation'] = torch.FloatTensor(flame_param['rotation'])[0]
    flame_param_tensor['neck_pose'] = torch.FloatTensor(flame_param['neck_pose'])[0]
    flame_param_tensor['jaw_pose'] = torch.FloatTensor(flame_param['jaw_pose'])[0]
    flame_param_tensor['eyes_pose'] = torch.FloatTensor(flame_param['eyes_pose'])[0]
    flame_param_tensor['translation'] = torch.FloatTensor(flame_param['translation'])[0]
    if teeth_bs is not None:
        flame_param_tensor['teeth_bs'] = torch.FloatTensor(teeth_bs)
        # flame_param_tensor['expr'] = torch.cat([flame_param_tensor['expr'], flame_param_tensor['teeth_bs']], dim=0)

    return flame_param_tensor


def prepare_motion_seqs(motion_seqs_dir, image_folder, save_root, fps,
                        bg_color, aspect_standard, enlarge_ratio,
                        render_image_res, need_mask, multiply=16, 
                        vis_motion=False, shape_param=None, test_sample=False,
                        cross_id=False, src_driven=["", ""], vis_rgb=False,
                        vis_frame_len: Optional[int] = None,
                        cycle_sample_mv_cam: bool = False, frames_per_loop: int = 128,
                        is_nersemble_cam: bool = False):
    if motion_seqs_dir is None:
        assert image_folder is not None
        motion_seqs_dir, image_folder = predict_motion_seqs_from_images(image_folder, save_root, fps)
    
    # source images
    rgbs = []
    c2ws, intrs, bg_colors = [], [], []
    flame_params = []

    # read shape_param
    if shape_param is None:
        print("using driven shape params")
        cor_flame_path = os.path.join(os.path.dirname(motion_seqs_dir), 'canonical_flame_param.npz')
        flame_p = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_p['shape'])

    transforms_json = os.path.join(os.path.dirname(motion_seqs_dir), f"transforms.json")
    with open(transforms_json) as fp:
        data = json.load(fp)  
    all_frames = data["frames"]
    all_frames = sorted(all_frames, key=lambda x: x["flame_param_path"])

    if vis_frame_len is not None:
        len_vis_seq = min(len(all_frames), vis_frame_len)
        all_frames = all_frames[:len_vis_seq]

    print(f"len vis motion_seq:{len(all_frames)}")
    frame_ids = np.array(list(range(len(all_frames))))
    frame_ids = frame_ids[0: 3000]
    if test_sample:
        print("sub sample 200 frames for testing.")
        sample_num = 50
        frame_ids = frame_ids[np.linspace(0, frame_ids.shape[0]-1, sample_num).astype(np.int32)]
        # frame_ids = frame_ids[0: 1000]
        print("sub sample ids:", frame_ids)

    teeth_bs_pth = os.path.join(os.path.dirname(motion_seqs_dir), "tracked_teeth_bs.npz")
    if os.path.exists(teeth_bs_pth):
        teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
    else:
        teeth_bs_lst = None

    print("loading motion seqs ...")
    for frame_id in tqdm(frame_ids, total=len(frame_ids)):
        frame_info = all_frames[frame_id]
        flame_path = os.path.join(os.path.dirname(motion_seqs_dir), frame_info["flame_param_path"])
        teeth_bs = teeth_bs_lst[frame_id] if (teeth_bs_lst is not None and len(teeth_bs_lst) > frame_id) else None
        flame_param = load_flame_params(flame_path, teeth_bs)

        c2w, intr = load_cam_pose(frame_info)

        if vis_rgb:
            rgb_path = os.path.join(os.path.dirname(motion_seqs_dir), frame_info["file_path"])
            mask_path = os.path.join(os.path.dirname(motion_seqs_dir), frame_info["fg_mask_path"])
            rgb, _, intr, _ = preprocess_image(
                rgb_path=rgb_path, mask_path=mask_path, intr=intr, pad_ratio=0, bg_color=bg_color,
                max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=enlarge_ratio,
                render_tgt_size=render_image_res, multiply=multiply, need_mask=need_mask,
                get_shape_param=False)
            rgbs.append(rgb)  # [1, 3, H, W]
        
        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intr)
        flame_params.append(flame_param)
    print("load motion seqs finished")

    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
    bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    rgbs = torch.cat(rgbs, dim=0).permute(0, 2, 3, 1) if vis_rgb else None

    flame_params_tmp = defaultdict(list)
    for flame in flame_params:
        for k, v in flame.items():
            flame_params_tmp[k].append(v)
    for k, v in flame_params_tmp.items():
        flame_params_tmp[k] = torch.stack(v)
    flame_params = flame_params_tmp

    if cycle_sample_mv_cam:
        print("sample mv cams to cycle loop cam trajectory ...")
        print(f"Before Sampling: c2ws shape: {c2ws.shape}, intrs shape: {intrs.shape}")
        c2ws, intrs, sampled_indices = interpolate_multiview_cameras(c2ws, intrs,frames_per_loop )
        print(f"After Sampling: c2ws shape: {c2ws.shape}, intrs shape: {intrs.shape}")
        bg_colors = bg_colors[sampled_indices, :]
        flame_params = {k:v[sampled_indices, :] for k, v in flame_params_tmp.items()}
        rgbs = rgbs[sampled_indices, :, :, :] if vis_rgb else None
    
    # ---------- TEMP MODIFY ----------
    SCREEN_3D_VER = False
    if SCREEN_3D_VER:
        with open('/nas5/to_zijian/transforms_lam_shangping.json') as fp:
            data = json.load(fp)
        temp_frames = data["frames"]
        temp_frames = sorted(temp_frames, key=lambda x: x['timestep_index'])
        temp_c2ws, temp_intrs = [], []
        for temp_frame in temp_frames:
            temp_c2w, temp_intr = load_cam_pose(temp_frame)
            temp_c2ws.append(temp_c2w)
            temp_intrs.append(temp_intr)
        temp_c2ws = torch.stack(temp_c2ws, dim=0)  # [11, 4, 4]
        temp_intrs = torch.stack(temp_intrs, dim=0)  # [11, 4, 4]
        n_f = flame_params['expr'].shape[0]
        n_v = temp_c2ws.shape[0]

        n_all = n_v * n_f
        bg_colors = bg_colors.unsqueeze(1).repeat(1, n_v, 1).reshape(n_all, -1)
        for k, v in flame_params.items():
            flame_params[k] = v.unsqueeze(1).repeat(1, n_v, 1).reshape(n_all, -1)
        c2ws = temp_c2ws.unsqueeze(0).repeat(n_f, 1, 1, 1).reshape(n_all, 4, 4)
        intrs = temp_intrs.unsqueeze(0).repeat(n_f, 1, 1, 1).reshape(n_all, 4, 4)
    # ---------- TEMP MODIFY ----------

    flame_params["betas"] = shape_param
    motion_render = render_flame_mesh(flame_params, intrs, c2ws, is_nersemble_cam) if vis_motion else None

    motion_seqs = {}
    motion_seqs["render_c2ws"] = c2ws.unsqueeze(0)
    motion_seqs["render_intrs"] = intrs.unsqueeze(0)
    motion_seqs["render_bg_colors"] = bg_colors.unsqueeze(0)
    motion_seqs["vis_motion_render"] = motion_render  # [N, H, W, 3] or None
    motion_seqs["vis_ori_rgbs"] = rgbs  # [N, H, W, 3] or None
    for k, v in flame_params.items():
        flame_params[k] = v.unsqueeze(0)
    motion_seqs["flame_params"] = flame_params
    return motion_seqs


def tile_images_to_square(images: List[torch.Tensor],
                          fill_color: float = 0.5,
                          padding: int = 2) -> torch.Tensor:
    """
    将任意数量的图像拼接成接近正方形的网格
    
    Args:
        images: 图像列表，每个图像shape为 (C, H, W) 或 (H, W) 或 (B, C, H, W)
        fill_color: 填充颜色，0-1之间的值，0.5为灰色
        padding: 图像之间的间距
    
    Returns:
        拼接后的图像tensor: shape -> (C, final_h, final_w), range -> [0, 1]
    """
    if not images:
        raise ValueError("图像列表不能为空")
    
    # 标准化图像格式
    processed_images = []
    for img in images:
        if img.dim() == 2:  # 灰度图 (H, W) -> (1, H, W)
            img = img.unsqueeze(0)
        elif img.dim() == 3:  # 彩色图 (C, H, W)
            pass
        elif img.dim() == 4:
            img = img.squeeze(0)
        else:
            raise ValueError(f"不支持的图像维度: {img.dim()}")
        processed_images.append(img)
    
    # 获取图像信息
    num_images = len(processed_images)
    channels = processed_images[0].shape[0]
    
    # 统一所有图像的尺寸（使用最大尺寸）
    max_h = max(img.shape[1] for img in processed_images)
    max_w = max(img.shape[2] for img in processed_images)
    
    # 将所有图像resize到统一尺寸
    unified_images = []
    for img in processed_images:
        c, h, w = img.shape
        # 创建填充后的图像
        padded_img = torch.full((c, max_h, max_w), fill_color, dtype=img.dtype, device=img.device)
        # 将原图像居中放置
        start_h = (max_h - h) // 2
        start_w = (max_w - w) // 2
        padded_img[:, start_h:start_h+h, start_w:start_w+w] = img
        unified_images.append(padded_img)
    
    # 计算最接近正方形的网格尺寸
    grid_size = math.ceil(math.sqrt(num_images))
    
    # 创建网格
    grid_h = grid_w = grid_size
    
    # 计算最终图像尺寸
    final_h = grid_h * max_h + (grid_h + 1) * padding
    final_w = grid_w * max_w + (grid_w + 1) * padding
    
    # 创建最终图像
    result = torch.full((channels, final_h, final_w), fill_color, 
                       dtype=unified_images[0].dtype, device=unified_images[0].device)
    
    # 填充图像到网格中
    for idx, img in enumerate(unified_images):
        row = idx // grid_w
        col = idx % grid_w
        
        # 计算位置
        start_h = padding + row * (max_h + padding)
        start_w = padding + col * (max_w + padding)
        
        result[:, start_h:start_h+max_h, start_w:start_w+max_w] = img
    
    return result


def pad_to_square(
        rgb: np.ndarray,
        mask: np.ndarray,
        bg_color: float,
        intrinsic: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    """
    将非方形的图像、蒙版和内参矩阵通过中心填充处理为方形。

    如果输入图像的高度和宽度不相等，该函数会计算所需的填充量，
    将图像和蒙版填充为边长等于原尺寸最大值的新方形。
    RGB图像使用指定的背景颜色进行填充，蒙版使用0.0进行填充。
    同时，函数会正确地更新相机内参矩阵中的主点(cx, cy)以匹配新的尺寸。

    参数:
        rgb (np.ndarray): 输入的RGB图像，形状为 (h, w, 3)，数值在 [0, 1] 范围内。
        mask (np.ndarray): 输入的蒙版，形状为 (h, w)。
        intrinsic (torch.Tensor): 4x4的相机内参矩阵。
        bg_color (float): 用于填充RGB图像背景的颜色值。

    返回:
        Tuple[np.ndarray, np.ndarray, torch.Tensor]: 一个包含以下三个元素的元组：
            - padded_rgb: 填充后的方形RGB图像，形状为 (a, a, 3)，其中 a = max(h, w)。
            - padded_mask: 填充后的方形蒙版，形状为 (a, a)。
            - new_intrinsic: 更新后的相机内参矩阵。
    """
    # 1. 获取原始尺寸
    h, w = rgb.shape[:2]
    
    # 2. 如果图像已经是方形，则无需处理，直接返回原始输入
    if h == w:
        return rgb, mask, intrinsic

    # 3. 计算目标尺寸和所需的填充量
    target_size = max(h, w)
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

    if h < w:
        # 如果宽度大于高度，则在垂直方向（上下）进行填充
        pad_total_h = w - h
        pad_top = pad_total_h // 2
        pad_bottom = pad_total_h - pad_top
    else: # w < h
        # 如果高度大于宽度，则在水平方向（左右）进行填充
        pad_total_w = h - w
        pad_left = pad_total_w // 2
        pad_right = pad_total_w - pad_left
        
    # 4. 更新内参矩阵
    # 填充会改变图像坐标系的原点，因此需要调整主点(cx, cy)
    # Focal length (fl_x, fl_y) 是相机的物理属性，与图像尺寸无关，保持不变。
    if intrinsic is not None:
        new_intrinsic = intrinsic.clone()  # 克隆一个新张量，避免修改原始输入
        new_intrinsic[0, 2] += pad_left   # cx = cx + 左侧填充量
        new_intrinsic[1, 2] += pad_top    # cy = cy + 顶部填充量
    else:
        new_intrinsic = None
    
    # 5. 对RGB图像和蒙版进行填充
    # np.pad 是一个非常高效的填充工具
    # 为RGB图像定义填充配置 (h, w, c) -> (上下填充, 左右填充, 通道不填充)
    rgb_padding_config = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    padded_rgb = np.pad(
        rgb,
        pad_width=rgb_padding_config,
        mode='constant',
        constant_values=bg_color  # 使用指定的背景色填充
    )
    
    # 为蒙版定义填充配置 (h, w) -> (上下填充, 左右填充)
    mask_padding_config = ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_mask = np.pad(
        mask,
        pad_width=mask_padding_config,
        mode='constant',
        constant_values=0.0  # 蒙版填充区域值为0
    )
    
    return padded_rgb, padded_mask, new_intrinsic
