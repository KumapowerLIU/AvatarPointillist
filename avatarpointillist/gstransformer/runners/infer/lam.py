# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. All rights reserved.
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
import traceback
import time
import torch
import os
import cv2
import numpy as np

from PIL import Image
from glob import glob
from tqdm.auto import tqdm
from safetensors.torch import load_file
# from accelerate.logging import get_logger

from .base_inferrer import Inferrer
from gstransformer.runners import REGISTRY_RUNNERS
# from lam.utils.logging import configure_logger
# from lam.utils.hf_hub import wrap_model_hub
from gstransformer.utils.config import parse_configs
from gstransformer.utils.build_model import build_model
from gstransformer.utils.video import add_audio_to_video, save_imgs_2_video_imageio, reshape_video, reshape_video_self,save_imgs_2_video
from gstransformer.utils.data_prepare import prepare_motion_seqs, preprocess_image, tile_images_to_square
# from tools.flame_tracking_single_image import FlameTrackingSingleImage


@REGISTRY_RUNNERS.register('infer.lam')
class LAMInferrer(Inferrer):
    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        
        # configure_logger(
        #     stream_level=self.cfg.logger.stream_level,
        #     log_level=self.cfg.logger.log_level)
        
        self.model = build_model(self.cfg, self.cfg.experiment.type).to(self.device)
        
        # self.flametracking = FlameTrackingSingleImage(
        #     output_dir='tracking_output',
        #     alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
        #     vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
        #     human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
        #     facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
        #     detect_iris_landmarks=True,
        #     args = self.cfg
        # )
        
        # hard code
        self.nersemble_motion_prefix = (
            'EMO-1', 'EMO-2', 'EMO-3', 'EMO-4', 'FREE', 'EXP-2', 'EXP-3', 'EXP-4', 'EXP-5', 'EXP-8', 'EXP-9'
        )
    
    def infer_single(self, image_path: str,
                     motion_seqs_dir, 
                     motion_img_dir,
                     motion_video_read_fps,
                     export_video: bool, 
                     export_mesh: bool, 
                     dump_tmp_dir:str,  # require by extracting motion seq from video, to save some results
                     dump_image_dir:str,
                     dump_video_path: str, 
                     dump_mesh_path: str,
                     gaga_track_type: str):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = self.cfg.render_fps
        aspect_standard = 1.0/1.0
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False
        save_ply = self.cfg.get("save_ply", False)  # False
        save_img = self.cfg.get("save_img", False)  # False
        rendered_bg = 1.
        ref_bg = 1.
        mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
        if ref_bg < 1.:
            if "VFHQ_TEST" in image_path:
                mask_path = image_path.replace("/VFHQ_TEST/", "/mask/").replace("/images/", "/mask/").replace(".png", ".jpg")
            else:
                mask_path = image_path.replace("/vfhq_test_nooffset_export/", "/mask/").replace("/images/", "/mask/").replace(".png", ".jpg")
        if not os.path.exists(mask_path):
            print("Warning: Mask path not exists:", mask_path)
            mask_path = None
        else:
            print("load mask from:", mask_path)

        if self.cfg.model.encoder_type == 'dinov2_fusion':
            PATCH_SIZE = 14
        elif self.cfg.model.encoder_type == 'dinov3_fusion':
            PATCH_SIZE = 16
        else:
            assert False, f"encoder_type: {self.cfg.model.encoder_type} not supported!"

        image, _, _, shape_param = preprocess_image(image_path, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=ref_bg, 
                                             max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                             render_tgt_size=source_size, multiply=PATCH_SIZE, need_mask=True, get_shape_param=True)
        # save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "refer_" + os.path.basename(image_path))
        vis_ref_img = (image[0].permute(1, 2 ,0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)
        # prepare motion seq
        test_sample=self.cfg.get("test_sample", False)
        # test_sample=True
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, motion_img_dir, save_root=dump_tmp_dir, fps=motion_video_read_fps,
                                            bg_color=rendered_bg, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1,0],
                                            render_image_res=render_size,  multiply=16, 
                                            need_mask=motion_img_need_mask, vis_motion=vis_motion, 
                                            shape_param=shape_param, test_sample=test_sample, cross_id=self.cfg.get("cross_id", False), src_driven=src_driven)

        # return

        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        start_time = time.time()
        device="cuda"
        dtype=torch.float32
        # dtype=torch.bfloat16
        self.model.to(dtype)
        print("start to inference...................")
        with torch.no_grad():
            # TODO check device and dtype
            res = self.model.infer_single_view(image.unsqueeze(0).to(device, dtype), None, None, 
                                               render_c2ws=motion_seq["render_c2ws"].to(device),
                                               render_intrs=motion_seq["render_intrs"].to(device),
                                               render_bg_colors=motion_seq["render_bg_colors"].to(device),
                                               flame_params={k:v.to(device) for k, v in motion_seq["flame_params"].items()})

        print(f"time elapsed: {time.time() - start_time}")
        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        only_pred = rgb
        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)
            import cv2
            vis_ref_img = np.tile(cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)[None, :, :, :], (rgb.shape[0], 1, 1, 1))
            blend_ratio = 0.7
            blend_res = ((1 -  blend_ratio) * rgb + blend_ratio * motion_seq["vis_motion_render"]).astype(np.uint8)
            # rgb = np.concatenate([rgb, motion_seq["vis_motion_render"], blend_res, vis_ref_img], axis=2)
            rgb = np.concatenate([vis_ref_img, rgb, motion_seq["vis_motion_render"]], axis=2)
            
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        save_imgs_2_video(rgb, dump_video_path, render_fps)
        base_vid = motion_seqs_dir.strip('/').split('/')[-1]
        audio_path = os.path.join(motion_seqs_dir, base_vid+".wav")
        dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
        add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
        if save_img and dump_image_dir is not None:
            for i in range(rgb.shape[0]):
                save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
                Image.fromarray(only_pred[i]).save(save_file)
                if save_ply and dump_mesh_path is not None:
                    res["3dgs"][i][0][0].save_ply(os.path.join(dump_image_dir, f"{i:04d}.ply"))

            dump_cano_dir = "./dumps/cano_gs/"
            if not os.path.exists(dump_cano_dir):
                os.system(f"mkdir -p {dump_cano_dir}")
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + ".ply")
            # res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=True, offset2xyz=False)
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + "_gs_offset.ply")
            res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=True, offset2xyz=True)
            # res['cano_gs_lst'][0].save_ply("tmp.ply", rgb2sh=False, offset2xyz=True)

            def save_color_points(points, colors, sv_pth, sv_fd="debug_vis/dataloader/"):
                points = points.squeeze().detach().cpu().numpy()
                colors = colors.squeeze().detach().cpu().numpy()
                sv_pth = os.path.join(sv_fd, sv_pth)
                if not os.path.exists(sv_fd):
                    os.system(f"mkdir -p {sv_fd}")
                with open(sv_pth, 'w') as of:
                    for point, color in zip(points, colors):
                        print('v', point[0], point[1], point[2], color[0], color[1], color[2], file=of)
 
            # save canonical color point clouds
            save_color_points(res['cano_gs_lst'][0].xyz, res["cano_gs_lst"][0].shs[:, 0, :], "framework_img.obj", sv_fd=dump_cano_dir) 

            # Export the template mesh to an OBJ file
            import trimesh
            vtxs = res['cano_gs_lst'][0].xyz - res['cano_gs_lst'][0].offset
            vtxs = vtxs.detach().cpu().numpy() 
            faces = self.model.renderer.flame_model.faces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vtxs, faces=faces)
            mesh.export(os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_shaped_mesh.obj'))

            # Export textured deformed mesh
            import lam.models.rendering.utils.mesh_utils as mesh_utils
            vtxs = res['cano_gs_lst'][0].xyz.detach().cpu()
            faces = self.model.renderer.flame_model.faces.detach().cpu()
            colors = res['cano_gs_lst'][0].shs.squeeze(1).detach().cpu()
            pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_textured_mesh.obj')
            print("Save textured mesh to:", pth)
            mesh_utils.save_obj(pth, vtxs, faces, textures=colors, texture_type="vertex")

    def infer(self):
        image_paths = []
        # hard code
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths = [self.cfg.image_input]
        else:
            # ids = sorted(os.listdir(self.cfg.image_input))
            # image_paths = [os.path.join(self.cfg.image_input, e, "images/00000_00.png") for e in ids]
            image_paths = glob(os.path.join(self.cfg.image_input, "*.jpg"))
            omit_prefix = self.cfg.image_input

        gaga_track_type = ""

        for image_path in tqdm(image_paths, disable=not self.accelerator.is_local_main_process):
            try:

                # preprocess input image: segmentation, flame params estimation
                return_code = self.flametracking.preprocess(image_path)
                assert (return_code == 0), "flametracking preprocess failed!"
                return_code = self.flametracking.optimize()
                assert (return_code == 0), "flametracking optimize failed!"
                return_code, output_dir = self.flametracking.export()
                assert (return_code == 0), "flametracking export failed!"

                image_path = os.path.join(output_dir, "images/00000_00.png")
                # mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")

                motion_seqs_dir = self.cfg.motion_seqs_dir
                print("motion_seqs_dir:", motion_seqs_dir)
                # prepare dump paths
                image_name = os.path.basename(image_path)
                uid = image_name.split('.')[0]
                subdir_path = os.path.dirname(image_path).replace(omit_prefix, '')
                subdir_path = subdir_path[1:] if subdir_path.startswith('/') else subdir_path
                # hard code
                subdir_path = gaga_track_type
                uid = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
                print("subdir_path and uid:", subdir_path, uid)
                dump_video_path = os.path.join(
                    self.cfg.video_dump,
                    subdir_path,
                    f'{uid}.mp4',
                )
                dump_image_dir = os.path.join(
                    self.cfg.image_dump,
                    subdir_path,
                    f'{uid}'
                )
                dump_tmp_dir = os.path.join(
                    self.cfg.image_dump,
                    subdir_path,
                    "tmp_res"
                )
                dump_mesh_path = os.path.join(
                    self.cfg.mesh_dump,
                    subdir_path,
                    # f'{uid}.ply',
                )
                os.makedirs(dump_image_dir, exist_ok=True)
                os.makedirs(dump_tmp_dir, exist_ok=True)
                os.makedirs(dump_mesh_path, exist_ok=True)

                # if os.path.exists(dump_video_path):
                #     print(f"skip:{image_path}")
                #     continue

                self.infer_single(
                    image_path,
                    motion_seqs_dir=motion_seqs_dir,
                    motion_img_dir=self.cfg.motion_img_dir,
                    motion_video_read_fps=self.cfg.motion_video_read_fps,
                    export_video=self.cfg.export_video,
                    export_mesh=self.cfg.export_mesh, 
                    dump_tmp_dir=dump_tmp_dir,
                    dump_image_dir=dump_image_dir,
                    dump_video_path=dump_video_path, 
                    dump_mesh_path=dump_mesh_path,
                    gaga_track_type=gaga_track_type
                    )
            except:
                traceback.print_exc()
    def _load_gs_npz(self, index_apth, lbs_shape_path, ar_feature_path ):
        data = np.load(index_apth, allow_pickle=True).item()
        xyz = data['xyz']
        binding = data['binding'] 
        
        binding = binding  

        ar_feaures = np.load( ar_feature_path ) 
        ar_feaures = ar_feaures.squeeze(0)[3:, :]
        seq_len = ar_feaures.shape[0] // 4
        assert seq_len == xyz.shape[0]
        sorted_xyz, sorted_binding = xyz, binding
        data_lbs_shape = np.load(lbs_shape_path)
        my_lbs_weights = torch.from_numpy(data_lbs_shape['my_lbs_weights']).float().unsqueeze(0)  
        my_shapedirs_up = torch.from_numpy(data_lbs_shape['my_shapedirs_up']).float().unsqueeze(0)  
        gs = {
        'gs_xyz': torch.from_numpy(sorted_xyz).float().unsqueeze(0),  # [1, N, 3]
        'gs_binding': torch.from_numpy(sorted_binding).long().unsqueeze(0),  # [1,N] or [1,N, k]
        'ar_features': torch.from_numpy(ar_feaures).float().unsqueeze(0),  # [1,4N, C]
        'my_lbs_weights': my_lbs_weights,  # [N, 104]
        'my_shapedirs_up': my_shapedirs_up,  # [N,
        }
        return gs
    def infer_nsb(self):
        assert self.cfg.image_input is not None, 'image_input is required'
        assert self.cfg.input_image_name is not None, 'input_image_name is required'
        assert self.cfg.gs_input is not None, 'gs_input is required'
        image_dir = self.cfg.image_input
        assert os.path.isdir(image_dir), "IMAGE_INPUT must be a directory"
        print("input_images_dir:", image_dir)

        # TODO: hard code here: .png
        input_num = len(glob(os.path.join(image_dir, "*.png")))
        print(f"input_num: {input_num}")

        motion_seqs_dir = self.cfg.motion_seqs_dir
        assert motion_seqs_dir.endswith('/'), "MOTION_SEQS_DIR must end with '/'"
        print("motion_seqs_dir:", motion_seqs_dir)

        is_nersemble_motion = False
        for prefix in self.nersemble_motion_prefix:
            if prefix in motion_seqs_dir:
                is_nersemble_motion = True
                break
        print(f"is_nersemble_motion: {is_nersemble_motion}")

        # prepare dump paths
        image_name = os.path.basename(os.path.normpath(image_dir))
        uid = image_name.split('_')[0] if '_' in image_name else image_name  # e.g., 106_EMO-4
        seq_motion_name = os.path.basename(os.path.normpath(motion_seqs_dir))
        seq_motion_name = seq_motion_name.split('_')[-1] if '_' in seq_motion_name else seq_motion_name  # e.g., 076_FREE
        print("uid and seq:", uid, seq_motion_name)

        dump_video_path = os.path.join(self.cfg.video_dump, uid, f'{uid}_{seq_motion_name}_input_{input_num}.mp4')
        dump_video_path_gt = os.path.join(self.cfg.video_dump, uid,  'gt')
        dump_video_path_gen = os.path.join(self.cfg.video_dump, uid,  'gen')
        dump_image_dir = os.path.join(self.cfg.image_dump, uid, f'{seq_motion_name}_input_{input_num}')
        dump_tmp_dir = os.path.join(self.cfg.image_dump, uid, "tmp_res")
        dump_mesh_path = os.path.join(self.cfg.mesh_dump, uid,f'{seq_motion_name}_input_{input_num}')
        os.makedirs(dump_image_dir, exist_ok=True)
        os.makedirs(dump_tmp_dir, exist_ok=True)
        os.makedirs(dump_mesh_path, exist_ok=True)
        os.makedirs(dump_video_path_gen, exist_ok=True)
        os.makedirs(dump_video_path_gt, exist_ok=True)
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        motion_img_dir=self.cfg.motion_img_dir
        motion_video_read_fps=self.cfg.motion_video_read_fps
        export_video=self.cfg.export_video
        export_mesh=self.cfg.export_mesh
        assert export_video or export_mesh, \
            'At least one of export_video or export_mesh should be True'

        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = self.cfg.render_fps
        print("render_fps", render_fps)
        aspect_standard = 1.0/1.0
        rendered_bg = 1.
        ref_bg = 1.

        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False
        vis_rgb = self.cfg.get("vis_rgb", False)  # False
        save_ply = self.cfg.get("save_ply", False)  # False
        save_img = self.cfg.get("save_img", False)  # False
        save_img_type = self.cfg.get("save_img_type", "ori")  # 'ori' or 'cat'
        vis_len = self.cfg.get("vis_len", None)
        frames_per_loop = self.cfg.get("frames_per_loop", 128)
        vis_blend_res = self.cfg.get("vis_blend_res", False)
        vis_ref = self.cfg.get("vis_ref", True)
        if is_nersemble_motion:
            cycle_sample_mv_cam = self.cfg.get("cycle_sample_mv_cam", False)
        else:
            cycle_sample_mv_cam = False

        if self.cfg.model.encoder_type == 'dinov2_fusion':
            PATCH_SIZE = 14
        elif self.cfg.model.encoder_type == 'dinov3_fusion':
            PATCH_SIZE = 16
        else:
            assert False, f"encoder_type: {self.cfg.model.encoder_type} not supported!"

        # image load & preprocess
        image_list = []
        name_img = self.cfg.input_image_name
        name_img = name_img if name_img.endswith('.png') else name_img + '.png'
        image_path = os.path.join(image_dir, name_img)
        # mask_path = image_path.replace("/inference_imgs/", "/motion_seqs/").replace(".jpg", ".png")
        image, _, _, _ = preprocess_image(
            image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=ref_bg,
            max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=PATCH_SIZE, need_mask=True, get_shape_param=False)
        image_list.append(image)

        # save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "refer_" + os.path.basename(image_path))
        vis_ref_img = (image[0].permute(1, 2 ,0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)
        image = torch.cat(image_list, dim=0)


        # get ar out as input
        index_path = os.path.join(self.cfg.gs_input,'new_point_data.npy')
        ar_feature_path = os.path.join(self.cfg.gs_input,'topk_1_topp_0.95_temp_0.0_embed.npy')
        lbs_shape_path = os.path.join(self.cfg.gs_input,'gs_interp.npz')

        gs_data =  self._load_gs_npz(index_path, lbs_shape_path, ar_feature_path)
        
        # tile ref images to square
        ref_img_total_path = os.path.join(dump_tmp_dir, "refer_img_total.png")
        vis_ref_img_total = tile_images_to_square(image_list)
        vis_ref_img_total = (vis_ref_img_total.permute(1, 2 ,0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img_total).save(ref_img_total_path)
        
        # shape_param load
        cor_flame_path = os.path.join(image_dir, 'canonical_flame_param.npz')
        flame_p = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_p['shape'])
        
        # prepare motion seq
        motion_seq = prepare_motion_seqs(
            motion_seqs_dir, image_folder=None, save_root=dump_tmp_dir,
            fps=motion_video_read_fps, bg_color=rendered_bg,
            aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1,0],
            render_image_res=render_size, multiply=16,
            need_mask=motion_img_need_mask, vis_motion=vis_motion,
            shape_param=shape_param, test_sample=self.cfg.test_sample,
            cross_id=self.cfg.get("cross_id", False),
            src_driven=[image_path.split('/')[-3], motion_seqs_dir.split('/')[-2]],
            vis_rgb=vis_rgb, vis_frame_len=vis_len, cycle_sample_mv_cam=cycle_sample_mv_cam,
            frames_per_loop=750, is_nersemble_cam=is_nersemble_motion)
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)


        # inference model
        device="cuda"
        dtype=torch.float32  # torch.bfloat16
        self.model.to(dtype)
        start_time = time.time()
        print("start to inference...................")
        with torch.no_grad():
            # ---------- TEMP MODIFY ----------
            SCREEN_3D_VER = False
            if SCREEN_3D_VER:
                vis_ref = False
                vis_motion = False
                cycle_sample_mv_cam = True
                
                rgbs = []
                chunk_size = 5
                n_batch = motion_seq["render_c2ws"].shape[1] // chunk_size
                ori_flame_param = motion_seq["flame_params"]
                betas = ori_flame_param["betas"].to(device)
                del ori_flame_param["betas"]
                for i in range(chunk_size):
                    render_c2ws = motion_seq["render_c2ws"][:, i * n_batch:(i + 1) * n_batch, ...].to(device)
                    render_intrs = motion_seq["render_intrs"][:, i * n_batch:(i + 1) * n_batch, ...].to(device)
                    render_bg_colors = motion_seq["render_bg_colors"][:, i * n_batch:(i + 1) * n_batch, ...].to(device)
                    flame_params = {k:v[:, i * n_batch:(i + 1) * n_batch, ...].to(device) for k, v in ori_flame_param.items()}
                    flame_params["betas"] = betas
                    res = self.model.infer_single_view(
                        gs_data['gs_xyz'].to(device, dtype),
                        gs_data['gs_binding'].to(device, torch.long),
                        gs_data['my_lbs_weights'].to(device, dtype), gs_data['my_shapedirs_up'].to(device, dtype),
                        image.unsqueeze(0).to(device, dtype), None, None,
                        render_c2ws=render_c2ws,
                        render_intrs=render_intrs,
                        render_bg_colors=render_bg_colors,
                        flame_params=flame_params,
                        render_h=810, render_w=1440, ar_features = gs_data['ar_features'].to(device, dtype)
                        )
                    rgbs.append(res["comp_rgb"].detach().cpu().numpy())
                rgb = np.concatenate(rgbs, axis=0)
            # ---------- TEMP MODIFY ----------
            else:
                res = self.model.infer_single_view(
                    gs_data['gs_xyz'].to(device, dtype),
                    gs_data['gs_binding'].to(device, torch.long),
                        gs_data['my_lbs_weights'].to(device, dtype), gs_data['my_shapedirs_up'].to(device, dtype),

                    image.unsqueeze(0).to(device, dtype), None, None,
                    render_c2ws=motion_seq["render_c2ws"].to(device),
                    render_intrs=motion_seq["render_intrs"].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"].to(device),
                    flame_params={k:v.to(device) for k, v in motion_seq["flame_params"].items()},
                    ar_features = gs_data['ar_features'].to(device, dtype)
                    )
                rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        print(f"time elapsed: {round(time.time() - start_time, 2)}s")


        # save video
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        ori_rgb = rgb
        grid_vid_list = []

        if vis_ref:
            vis_ref_img_total = cv2.resize(vis_ref_img_total, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)
            vis_ref_img_total = np.tile(vis_ref_img_total[None, :, :, :], (rgb.shape[0], 1, 1, 1))
            grid_vid_list.append(vis_ref_img_total)

        grid_vid_list.append(rgb)

        if vis_rgb and not cycle_sample_mv_cam:
            ori_driven_imgs = (motion_seq['vis_ori_rgbs'].cpu().detach().numpy() * 255).astype(np.uint8)
            grid_vid_list.append(ori_driven_imgs)
            print(ori_driven_imgs.shape)
        if vis_motion:
            rendered_mesh = motion_seq["vis_motion_render"]
            if vis_blend_res:
                blend_ratio = 0.7
                blend_res = ((1 -  blend_ratio) * rgb + blend_ratio * rendered_mesh).astype(np.uint8)
                grid_vid_list.append(blend_res)
            grid_vid_list.append(rendered_mesh)
        rgb = np.concatenate(grid_vid_list, axis=2)
        print(rgb.shape)
        if is_nersemble_motion and not cycle_sample_mv_cam:
            h_w_grid = (8, 2) if vis_rgb and vis_motion else (4, 4)
            rgb = reshape_video(rgb, N_views=16, h_w_grid=h_w_grid)
            rgb_gt_video_list = reshape_video_self(ori_driven_imgs, N_views=16, h_w_grid=(4, 4))
            ori_rgb_video_list = reshape_video_self(ori_rgb, N_views=16, h_w_grid=(4, 4))

        print()
        if SCREEN_3D_VER:
            rgb = reshape_video(rgb, N_views=50, h_w_grid=(10, 5))
        save_imgs_2_video_imageio(rgb, dump_video_path, render_fps)
        for i in range(len(ori_rgb_video_list)):
            ori_rgb_video = ori_rgb_video_list[i]
            rgb_gt_video = rgb_gt_video_list[i]
            dump_video_path_gen_each_view = os.path.join(dump_video_path_gen, f"gen_view_{str(i).zfill(2)}.mp4")
            dump_video_path_gt_each_view = os.path.join(dump_video_path_gt, f"gen_view_{str(i).zfill(2)}.mp4")
            save_imgs_2_video_imageio(ori_rgb_video, dump_video_path_gen_each_view, render_fps)
            save_imgs_2_video_imageio(rgb_gt_video,dump_video_path_gt_each_view, render_fps)

        # save images/mesh ...
        if save_img and dump_image_dir is not None:
            assert save_img_type in ["ori", "cat"]
            save_rgb = rgb if save_img_type == 'cat' else ori_rgb
            for i in range(save_rgb.shape[0]):
                save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
                Image.fromarray(save_rgb[i]).save(save_file)

                if save_ply and dump_mesh_path is not None:
                    res["3dgs"][i][0][0].save_ply(os.path.join(dump_mesh_path, f"{i:04d}.ply"))

            SAVE_CANO_GS = False
            if SAVE_CANO_GS:
                dump_cano_dir = "./dumps/cano_gs/"
                if not os.path.exists(dump_cano_dir):
                    os.system(f"mkdir -p {dump_cano_dir}")
                cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(os.path.normpath(dump_image_dir)) + ".ply")
                # res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=True, offset2xyz=False)
                cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(os.path.normpath(dump_image_dir)) + "_gs_offset.ply")
                res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=True, offset2xyz=True)
                # res['cano_gs_lst'][0].save_ply("tmp.ply", rgb2sh=False, offset2xyz=True)

                def save_color_points(points, colors, sv_pth, sv_fd="debug_vis/dataloader/"):
                    points = points.squeeze().detach().cpu().numpy()
                    colors = colors.squeeze().detach().cpu().numpy()
                    sv_pth = os.path.join(sv_fd, sv_pth)
                    if not os.path.exists(sv_fd):
                        os.system(f"mkdir -p {sv_fd}")
                    with open(sv_pth, 'w') as of:
                        for point, color in zip(points, colors):
                            print('v', point[0], point[1], point[2], color[0], color[1], color[2], file=of)
    
                # save canonical color point clouds
                save_color_points(res['cano_gs_lst'][0].xyz, res["cano_gs_lst"][0].shs[:, 0, :], "framework_img.obj", sv_fd=dump_cano_dir) 

                # Export the template mesh to an OBJ file
                import trimesh
                vtxs = res['cano_gs_lst'][0].xyz - res['cano_gs_lst'][0].offset
                vtxs = vtxs.detach().cpu().numpy() 
                faces = self.model.renderer.flame_model.faces.detach().cpu().numpy()
                mesh = trimesh.Trimesh(vertices=vtxs, faces=faces)
                mesh.export(os.path.join(dump_cano_dir, os.path.basename(os.path.normpath(dump_image_dir)) + '_shaped_mesh.obj'))

                # Export textured deformed mesh
                import lam.models.rendering.utils.mesh_utils as mesh_utils
                vtxs = res['cano_gs_lst'][0].xyz.detach().cpu()
                faces = self.model.renderer.flame_model.faces.detach().cpu()
                colors = res['cano_gs_lst'][0].shs.squeeze(1).detach().cpu()
                pth = os.path.join(dump_cano_dir, os.path.basename(os.path.normpath(dump_image_dir)) + '_textured_mesh.obj')
                print("Save textured mesh to:", pth)
                mesh_utils.save_obj(pth, vtxs, faces, textures=colors, texture_type="vertex")
