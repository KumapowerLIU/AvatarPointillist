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

import os
import time
import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from einops import rearrange, repeat
# from diffusers import AutoencoderKL
from .transformer import TransformerDecoder
from .rendering.gs_renderer import GS3DRenderer, PointEmbed, BindingEmbed
from diffusers import AutoencoderKL
from diffusers.utils import is_torch_version
import torchvision.transforms.functional as F
logger = get_logger(__name__)

class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s=self.shorten_factor)
        return self.proj(x)

class ComplexMLP(nn.Module):
    """
    一个包含 隐藏层、非线性激活、LayerNorm 和 Dropout 的复杂 MLP 模块。
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        初始化一个两层的 MLP。

        Args:
            input_dim (int): 输入特征的维度。
            output_dim (int): 最终输出特征的维度 (例如 d_model)。
            hidden_dim (int): 中间隐藏层的维度。
            dropout (float): Dropout 的比率。
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # GELU 是 Transformer 中常用的、性能优秀的激活函数
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：Linear -> GELU -> LayerNorm -> Dropout -> Linear
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.layernorm(x) # 增加稳定性
        x = self.fc2(x)
        return x
class ModelLAM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self,
                 transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 transformer_type="cond",
                 tf_grad_ckpt=False,
                 encoder_grad_ckpt=False,
                 encoder_freeze: bool = True, encoder_type: str = 'dino',
                 encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768,
                 num_pcl: int=2048, pcl_dim: int=512,
                 human_model_path="./model_zoo/human_parametric_models",
                 flame_subdivide_num=2,
                 flame_type="flame",
                 gs_query_dim=None,
                 gs_use_rgb=False,
                 gs_sh=3,
                 gs_mlp_network_config=None,
                 gs_xyz_offset_max_step=1.8 / 32,
                 gs_clip_scaling=0.2,
                 shape_param_dim=100,
                 expr_param_dim=50,
                 fix_opacity=False,
                 fix_rotation=False,
                 flame_scale=1.0,
                 use_binding_as_condion=False,
                 use_ar_embedding_as_condition=False,
                 use_gsa_render= False,
                 only_ar_feature = False,
                 use_vae = False,
                 vae_path = None,
                 fix_encoder = False,
                 **kwargs,
                 ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        self.use_vae = use_vae
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.conf_use_pred_img = False
        self.conf_cat_feat = False and self.conf_use_pred_img  # True # False
        self.face_nums = 101444
        # modules
        # image encoder
        self.only_ar_feature = only_ar_feature
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=encoder_feat_dim,
            pretrained=kwargs.get("encoder_pretrained", True),
        )
        self.fix_encoder = fix_encoder
        if fix_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if self.use_vae:
            vae = AutoencoderKL.from_pretrained(
                vae_path
            )
            vae.eval()
            for param in vae.parameters():
                param.requires_grad = False
            self.vae = vae
            self.vae_nn = ComplexMLP(input_dim=4, output_dim=pcl_dim, hidden_dim=128)

        if use_ar_embedding_as_condition and use_binding_as_condion:
            self.downsample_ar = LinearDownsample(dim=1024, shorten_factor=4)
            self.fusion_layer = nn.Linear(1024+ pcl_dim + pcl_dim, pcl_dim)
        elif use_ar_embedding_as_condition:
            self.downsample_ar = LinearDownsample(dim=1024, shorten_factor=4)
            self.fusion_layer = nn.Linear(1024+ pcl_dim, pcl_dim)
        elif use_binding_as_condion:
            self.downsample_ar = None
            self.fusion_layer = nn.Linear(pcl_dim + pcl_dim, pcl_dim)
        else:
            self.downsample_ar = None
            self.fusion_layer =  None



        # learnable points embedding
        skip_decoder = False
        self.latent_query_points_type = kwargs.get("latent_query_points_type", "e2e_flame")
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl
            self.pcl_embeddings = nn.Embedding(num_pcl , pcl_dim)
        elif self.latent_query_points_type.startswith("flame"):
            latent_query_points_file = os.path.join(human_model_path, "flame_points", f"{self.latent_query_points_type}.npy")
            pcl_embeddings = torch.from_numpy(np.load(latent_query_points_file)).float()
            print(f"==========load flame points:{latent_query_points_file}, shape:{pcl_embeddings.shape}")
            self.register_buffer("pcl_embeddings", pcl_embeddings)
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        elif self.latent_query_points_type.startswith("e2e_flame"):
            skip_decoder = True
            self.pcl_embed = PointEmbed(dim=pcl_dim)
            if use_binding_as_condion:
                self.index_embeddings = BindingEmbed(self.face_nums , pcl_dim)
        else:
            raise NotImplementedError
        print("==="*16*3, f"\nskip_decoder: {skip_decoder}", "\n"+"==="*16*3)
        # transformer
        self.transformer = TransformerDecoder(
            block_type=transformer_type,
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim, cond_dim=encoder_feat_dim, mod_dim=None,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.use_binding_as_condion = use_binding_as_condion
        self.use_ar_embedding_as_condition =use_ar_embedding_as_condition

        # renderer
        self.renderer = GS3DRenderer(human_model_path=human_model_path,
                                     subdivide_num=flame_subdivide_num,
                                     smpl_type=flame_type,
                                     feat_dim=transformer_dim,
                                     query_dim=gs_query_dim,
                                     use_rgb=gs_use_rgb,
                                     sh_degree=gs_sh,
                                     mlp_network_config=gs_mlp_network_config,
                                     xyz_offset_max_step=gs_xyz_offset_max_step,
                                     clip_scaling=gs_clip_scaling,
                                     scale_sphere=kwargs.get("scale_sphere", False),
                                     shape_param_dim=shape_param_dim,
                                     expr_param_dim=expr_param_dim,
                                     fix_opacity=fix_opacity,
                                     fix_rotation=fix_rotation,
                                     skip_decoder=skip_decoder,
                                     decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
                                     gradient_checkpointing=self.gradient_checkpointing,
                                     add_teeth=kwargs.get("add_teeth", True),
                                     teeth_bs_flag=kwargs.get("teeth_bs_flag", False),
                                     oral_mesh_flag=kwargs.get("oral_mesh_flag", False),
                                     use_mesh_shading=kwargs.get('use_mesh_shading', False),
                                     render_rgb=kwargs.get("render_rgb", True),
                                     use_gsa_render = use_gsa_render
                                     )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight

    @staticmethod
    def _encoder_fn(encoder_type: str):
        from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
        return Dinov2FusionWrapper

    def forward_transformer(self, image_feats, camera_embeddings, query_points, query_feats=None):
        # assert image_feats.shape[0] == camera_embeddings.shape[0], \
        #     "Batch size mismatch for image_feats and camera_embeddings!"
        B = image_feats.shape[0]
        if self.only_ar_feature:
            x = query_feats
        elif self.latent_query_points_type == "embedding":
            range_ = torch.arange(self.num_pcl, device=image_feats.device)
            x =  self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1)) # [B, L, D]

        elif self.latent_query_points_type.startswith("flame"):
            x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat((B, 1, 1)) # [B, L, D]

        elif self.latent_query_points_type.startswith("e2e_flame"):
            x = self.pcl_embed(query_points) # [B, L, D]

        x = x.to(image_feats.dtype)
        if query_feats is not None and not self.only_ar_feature:
            x = self.fusion_layer(torch.cat([x, query_feats], dim=-1))
            # x = x + query_feats.to(image_feats.dtype)
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )  # [B, L, D]
        # x = x.to(image_feats.dtype)
        return x

    def forward_encode_image(self, image):
        # encode image
        if self.training and self.encoder_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            if self.fix_encoder:
                with torch.no_grad():
                    image_feats = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.encoder),
                        image,
                        **ckpt_kwargs,
                    )
                    if self.use_vae:
                        image_for_vae = F.resize(image, [512, 512], antialias=True)
                        image_for_vae_normalized = image_for_vae * 2.0 - 1.0
                        posterior = self.vae.encode(image_for_vae_normalized).latent_dist
                        image_vae_feats = posterior.mean
                        image_vae_feats = self.vae_nn(image_vae_feats.flatten(2).permute(0, 2, 1))
                        image_feats = torch.cat([image_feats, image_vae_feats], dim=1)
            else:
                image_feats = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.encoder),
                        image,
                        **ckpt_kwargs,
                    )
        else:
            if self.fix_encoder:
                with torch.no_grad():
                    image_feats = self.encoder(image)
                    if self.use_vae:
                        image_for_vae = F.resize(image, [512, 512], antialias=True)
                        image_for_vae_normalized = image_for_vae * 2.0 - 1.0
                        posterior = self.vae.encode(image_for_vae_normalized).latent_dist
                        image_vae_feats = posterior.mean
                        image_vae_feats = self.vae_nn(image_vae_feats.flatten(2).permute(0, 2, 1))
                        image_feats = torch.cat([image_feats, image_vae_feats], dim=1)
            else:
                image_feats = self.encoder(image)

        return image_feats

    @torch.compile
    def forward_latent_points(self, image, camera, query_points=None, query_binding= None, ar_features_input=None, additional_features=None):
        # image: [B, C_img, H_img, W_img]
        # camera: [B, D_cam_raw]
        B = image.shape[0]

        # encode image
        image_feats = self.forward_encode_image(image)

        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        if additional_features is not None and len(additional_features.keys()) > 0:
            seqlen = query_points.shape[1]
            binding_seqlen = additional_features['query_binding'].shape[1]
            assert seqlen == binding_seqlen, f"Sequence length mismatch for query_points and query_binding: {seqlen} vs {binding_seqlen}"
            # expand to
            query_feats =additional_features['query_binding']
        else:
            query_feats = None

        if self.use_binding_as_condion and self.use_ar_embedding_as_condition:
            seqlen = query_points.shape[1]
            binding_seqlen = query_binding.shape[1]
            assert seqlen == binding_seqlen, f"Sequence length mismatch for query_points and query_binding: {seqlen} vs {binding_seqlen}"
            query_feats_binding = self.index_embeddings(query_binding)
            query_feats = torch.cat([query_feats_binding, ar_features_input], dim=-1)
        elif self.use_binding_as_condion:
            seqlen = query_points.shape[1]
            binding_seqlen = query_binding.shape[1]
            assert seqlen == binding_seqlen, f"Sequence length mismatch for query_points and query_binding: {seqlen} vs {binding_seqlen}"
            query_feats = self.index_embeddings(query_binding)
        elif self.use_ar_embedding_as_condition:
            query_feats = ar_features_input



        # # embed camera
        # camera_embeddings = self.camera_embedder(camera)
        # assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
        #     f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating latent points
        tokens = self.forward_transformer(image_feats, camera_embeddings=None, query_points=query_points, query_feats=query_feats)

        return tokens, image_feats

    def forward(self, query_points, query_binding, my_lbs_weights,  my_shapedirs_up, image, source_c2ws, source_intrs, render_c2ws, render_intrs, render_bg_colors, flame_params, source_flame_params=None, render_images=None, data=None, ar_features=None):
        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # flame_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        if self.use_ar_embedding_as_condition:
            assert ar_features is not None, "ar_features should not be None when use_ar_embedding_as_condition is True"
            ar_features = ar_features.to(dtype=torch.float32)
            ar_features = self.downsample_ar(ar_features)

        assert image.shape[0] == render_c2ws.shape[0], "Batch size mismatch for image and render_c2ws"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        assert image.shape[0] == flame_params["betas"].shape[0], "Batch size mismatch for image and flame_params"
        assert image.shape[0] == flame_params["expr"].shape[0], "Batch size mismatch for image and flame_params"
        # print(query_points.shape,query_points.dtype, query_points.max(), query_points.min() )
        query_points_orginal = query_points.clone()
        original_length = query_points_orginal.shape[1]
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
        query_points = query_points.to(dtype=torch.float32)
        # ###################################
        # if self.latent_query_points_type.startswith("e2e_flame"):
        #     query_points, flame_params = self.renderer.get_query_points(flame_params,
        #                                                                 device=image.device)
        #     # query_points_clone = query_points.clone()
        # # 離譜操作 我服了

        # # print(query_points.shape,query_points.dtype, query_points.max(), query_points.min())
        # query_points = query_points[:, :original_length+1, :]
        # # # query_points_clone[:, -1, :] =  query_points[:, -1, :]
        # query_points[:, :original_length, :] =  query_points_orginal
        # # # query_points = query_points_clone
        # # print(query_points.shape,query_points.dtype, query_points.max(), query_points.min())
        # # additional_features = {"query_binding": query_binding}
        # last_row = query_binding[:, -1:]         # 取最後一行，形狀仍是 (1, 1)
        # query_binding = torch.cat([query_binding.clone(), last_row], dim=1)
        # ###################################




        additional_features = {}
        if self.use_binding_as_condion:
            query_binding_input = query_binding
        else:
            query_binding_input = None
        if self.use_ar_embedding_as_condition:
            ar_features_input = ar_features
        else:
            ar_features_input = None
        latent_points, image_feats = self.forward_latent_points(image[:, 0], camera=None, query_points=query_points,query_binding=query_binding_input, ar_features_input=ar_features_input, additional_features=additional_features)  # [B, N, C]


        additional_features.update({
            "image_feats": image_feats, "image": image[:, 0],
        })
        image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))
        additional_features["image_feats_bchw"] = image_feats_bchw
        # render target views
        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       c2w=render_c2ws,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=additional_features,
                                       binding = query_binding,
                                        my_lbs_weights = my_lbs_weights,  my_shapedirs_up =my_shapedirs_up,
        )

        N, M = render_c2ws.shape[:2]

        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"
        return {
            'latent_points': latent_points,
            **render_results,
        }

    @torch.no_grad()
    def infer_single_view(self, query_points, query_binding, my_lbs_weights, my_shapedirs_up, image, source_c2ws, source_intrs, render_c2ws,
                          render_intrs, render_bg_colors, flame_params, ar_features=None):
        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # flame_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        assert image.shape[0] == render_c2ws.shape[0], "Batch size mismatch for image and render_c2ws"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        assert image.shape[0] == flame_params["betas"].shape[0], "Batch size mismatch for image and flame_params"
        assert image.shape[0] == flame_params["expr"].shape[0], "Batch size mismatch for image and flame_params"
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
        assert image.shape[0] == 1
        num_views = render_c2ws.shape[1]
        additional_features = {}
        if self.use_ar_embedding_as_condition:
            assert ar_features is not None, "ar_features should not be None when use_ar_embedding_as_condition is True"
            ar_features = ar_features.to(dtype=torch.float32)
            ar_features = self.downsample_ar(ar_features)
        if self.use_binding_as_condion:
            query_binding_input = query_binding
        else:
            query_binding_input = None
        if self.use_ar_embedding_as_condition:
            ar_features_input = ar_features
        else:
            ar_features_input = None


        latent_points, image_feats = self.forward_latent_points(image[:, 0], camera=None, query_points=query_points,query_binding=query_binding_input, ar_features_input=ar_features_input, additional_features=additional_features)  # [B, N, C]
        image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))

        gs_model_list, query_points, flame_params, _ = self.renderer.forward_gs(gs_hidden_features=latent_points,
                                                query_points=query_points,
                                                flame_data=flame_params,
                                                additional_features={"image_feats": image_feats, "image": image[:, 0], "image_feats_bchw": image_feats_bchw})

        render_res_list = []
        for view_idx in range(num_views):
            render_res = self.renderer.forward_animate_gs(gs_model_list,
                                                          query_points,
                                                          self.renderer.get_single_view_smpl_data(flame_params, view_idx),
                                                          render_c2ws[:, view_idx:view_idx+1],
                                                          render_intrs[:, view_idx:view_idx+1],
                                                          render_h,
                                                          render_w,
                                                          render_bg_colors[:, view_idx:view_idx+1],
                                                          query_binding, my_lbs_weights = my_lbs_weights,  my_shapedirs_up =my_shapedirs_up,


                                                          )
            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            # print(f"out key:{k}")
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(0, 2, 3, 1)  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
            else:
                out[k] = v
        out['cano_gs_lst'] = gs_model_list
        return out
