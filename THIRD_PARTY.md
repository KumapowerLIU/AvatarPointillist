# Source and model notices

This branch preserves the notices supplied with the research source. It does not assert a new blanket license over the project or grant rights to third-party model assets.

| Component | Evidence / upstream |
|---|---|
| Custom AR Transformer | Headers credit lucidrains/x-transformers and NVIDIA ODISE. Refer to [x-transformers](https://github.com/lucidrains/x-transformers/blob/main/LICENSE) and [ODISE](https://github.com/NVlabs/ODISE/blob/main/LICENSE). |
| Gaussian decoder / LAM / OpenLRM-derived modules | Existing Apache-2.0 headers credit Alibaba 3DAIGC and Zexin He. Preserve the headers and upstream notices. |
| DINOv2 vendored code | Apache-2.0 headers; [upstream license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE). |
| Michelangelo | Supplied license at `point_encoder/miche/LICENSE`; [upstream](https://github.com/NeuralCarver/Michelangelo). Standalone unused CLIP/diffusion features have additional dependencies. |
| GaussianAvatars | `gaussian_avatar/LICENSE.md` (CC BY-NC-SA 4.0) and `LICENSE_GS.md`. |
| Differentiable Gaussian rasterizer | `gaussian_avatar/submodules/diff-gaussian-rasterization/LICENSE.md`; INRIA non-commercial research/evaluation terms in source. Modified CUDA source is retained. |
| FLAME / SMPL-derived code and assets | Existing MPG notices in the FLAME/LBS modules; obtain model permissions through [FLAME](https://flame.is.tue.mpg.de/). Raw FLAME model assets are not included in the public Git branch. |
| GLM | Headers missing from supplied archive; optional install script retrieves tag 0.9.9.8 from [g-truc/glm](https://github.com/g-truc/glm). Preserve its license when fetched. |
| Foreground matting utility | Existing Microsoft MIT notice; ONNX weights were not supplied. |

The full original Gaussian decoder checkpoint contains FLAME buffers (`renderer.flame_model*`). The private Hugging Face archive preserves these bytes and optimizer state for the owner. Public model redistribution has not been resolved by this code review. Do not describe the entire archive as a single MIT/Apache model.

`LEGAL.md` is an original language-precedence notice for comments; it is not a project-wide distribution license. The project owner should select appropriate terms for original code before calling this a complete public open-source release.
