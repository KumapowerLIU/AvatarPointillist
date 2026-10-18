# Setup and supported entry points

This branch contains a **static-review release candidate**, not a runtime-validated environment. No training, inference, CUDA build, or project import was executed during review. Commands below are instructions for a Linux CUDA machine; they were not run as part of the review.

## Environment

Use Python 3.10 or 3.11, an NVIDIA GPU, a compatible CUDA toolkit and a compiler. Install a matching PyTorch/torchvision CUDA build from [PyTorch](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
```

Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) against the same PyTorch/CUDA build. Build the bundled rasterizer:

```bash
bash scripts/install_rasterizer.sh
```

The script fetches the missing GLM headers at tag `0.9.9.8` and builds the existing CUDA source. It does not replace the modified rasterizer with a different upstream implementation. The old Michelangelo `requirements.txt` is an environment dump with conflicting historical versions; do not install it over the root environment. Root requirements are an inferred candidate set, **not a tested lockfile**. In particular, the custom Transformer depends on internal `x-transformers` APIs; a working version still needs to be pinned after runtime validation.

Gaussian avatar fitting under `gaussian_avatar/` is a separate upstream workflow. It also needs `simple-knn` (absent from the supplied package) and its own optional viewer/tracking dependencies. Refer to its existing README. The supported release paths below do not import its fitting CLI.

## Model files

The owner archive is [KumaPower/AvatarPointillist](https://huggingface.co/KumaPower/AvatarPointillist). It is **private**, because it contains original training state and FLAME assets/buffers. An authorized account and upstream asset permissions are required. Authenticate locally without putting tokens in source files:

```bash
hf auth login
python scripts/download_models.py
```

This downloads the AR checkpoint, shape VAE, DINOv2 base weights/configs, Gaussian decoder and the model assets under their original relative paths. Use `--include-training-state` to also download the decoder optimizer/random state. For reproducibility, pass a Hub commit SHA with `--revision`. The SHA-256 inventory is in `models_manifest.json`.

The complete 34-file archive was verified at revision `2b9e67ed5ec157a606504f37541f497d4ffb0378`. To download that exact version:

```bash
python scripts/download_models.py --revision 2b9e67ed5ec157a606504f37541f497d4ffb0378
```


The Gaussian decoder checkpoint already contains its larger DINOv2 encoder; inference no longer downloads an unrelated copy before restoring it. The AR encoder separately uses `pretrain_model/dinov2-base/`.

The supplied package **does not contain** `foreground-segmentation-model-vitl16_384.onnx`, LPIPS/AlexNet training weights, or an automatic portrait-to-FLAME tracking model. RGB matting is now loaded only when requested. For RGB images that need background removal, supply the ONNX file and set `AVATAR_MATTING_MODEL`; alternatively provide an RGBA image or explicit mask. No replacement model has been invented or silently downloaded during review.

## Geometry inference

Run all commands from the repository root:

```bash
python test_ar.py --config_dir configs/ar_inference.yaml --file_path_test configs/ar_demo.json --output_dir outputs/ar
```

The manifest maps IDs to `img_path` and `flame_path`. A legacy one-item list containing that mapping is also accepted. The FLAME `.npz` must contain tracked shape coefficients. Gaussian ground-truth PLY files are no longer needed. Four BOS tokens initialize generation; the old 4,000-token ground-truth prefix has been removed.

Each sample gets its own directory under `outputs/ar/<sample-id>/`, containing `new_point_data.npy`, `gs_interp.npz`, token IDs, predictor embeddings, a PLY, and `generation.json`. EOS is allowed only between complete four-token points. `--max_length` is the maximum **new-token** count and must be divisible by four; `--window_size` controls the attention window. Greedy generation (`--temp 0`) ignores sampling randomness; positive temperature applies both top-k and top-p filtering. Compilation is opt-in with `--compile`.

## Gaussian decoding / animation

```bash
bash avatarpointillist/scripts_gstrans/inference.sh
```

This uses the `076_FREE` portrait, newly generated AR output, and a compact 32-view/two-timestamp motion example. It is an input-format example, not an evaluation. The provided RGB portraits are treated as already prepared (`source_need_mask=false`); enable source masking with an available matting model when using raw RGB portraits. The compact motion omits target RGB images, so `vis_rgb=false` and `vis_motion=false` are deliberate defaults. Outputs go to `outputs/decoder/`.

The shell script accepts, in order: config, checkpoint directory, portrait directory, motion directory, portrait filename, AR output directory. The direct entry point is:

```bash
python -m avatarpointillist.gstransformer.launch infer.lam --config avatarpointillist/configs/gaussian-decoder-testing.yaml model_name=pretrain_model/gaussian_decoder image_input=toy_data/infer_img/076_FREE input_image_name=00026_10.png gs_input=outputs/ar/076_FREE motion_seqs_dir=toy_data/motion_demo/076_FREE/ source_need_mask=false export_video=true export_mesh=false test_sample=false motion_img_dir=null motion_img_need_mask=false render_fps=20 vis_rgb=false vis_motion=false
```

The decoder's binding-conditioning switch is enabled to match the supplied `[1024, 3072]` fusion weight and binding embedding. Checkpoints now load strictly and run in evaluation mode. This makes configuration errors visible; it is not a guarantee of numerical/visual correctness.

## Training

Training data are not provided. Configure the example manifests and dataset split lists for your own tracked multi-view data:

```bash
accelerate launch train_ar.py --config_dir configs/humanar_training_cluster.yaml --checkpoint_dir outputs/ar_checkpoints
bash avatarpointillist/scripts_gstrans/exp/train_lam.sh
```

AR training expects fitted `point_cloud.ply` (with binding properties), `flame_param.npz` in each `base_path`, and a tracked sequence/image directory in `img_path` (including `sequences_trainval.txt` as expected by the dataset reader). Decoder training expects the AR exports plus the tracked multi-view image dataset and split lists. See the readers for the exact schema.

AR batches pad variable token lengths with the ignored loss label. Its dataset performs CUDA work and therefore uses `num_workers=0`. Decoder defaults to batch size one because early-EOS point counts vary and its attention/rendering path has no ragged-point batch mask. Larger decoder batches require equal point counts or a separately validated batching implementation.

AR checkpoints store batch position, global step and scaler state; resumes fail visibly if loading fails. Old archives lack some of that metadata. Neither pipeline currently promises bitwise-identical random-number/data augmentation replay after resume. The corrected pixel-loss coefficient and generation semantics may change results relative to historical experiments; retraining/evaluation comparisons remain outstanding.
