import os
import argparse
from omegaconf import OmegaConf


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, unknown = parser.parse_known_args()

    assert args.config is not None, "config is required"
    cfg = OmegaConf.load(args.config)
    if 'base_config' in cfg:
        base_cfg = OmegaConf.load(cfg.base_config)
        del cfg['base_config']
        cfg = OmegaConf.merge(base_cfg, cfg)

    cli_cfg = OmegaConf.from_cli(unknown)
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # hard code
    cfg.source_size = cfg.data.dataset.source_image_res
    cfg.render_size = cfg.data.dataset.render_image.high
    _relative_path = os.path.join(cfg.experiment.parent, cfg.experiment.child)

    cfg.save_tmp_dump = os.path.join('/nas5/liuhongyu/AR_Results/baseline/ours/vfhq/dumps', 'save_tmp', _relative_path)
    cfg.image_dump = os.path.join('/nas5/liuhongyu/AR_Results/baseline/ours/vfhq/dumps', 'images', _relative_path)
    cfg.video_dump = os.path.join('/nas5/liuhongyu/AR_Results/baseline/ours/vfhq/dumps', 'videos', _relative_path)
    cfg.mesh_dump = os.path.join('/nas5/liuhongyu/AR_Results/baseline/ours/vfhq/dumps', 'meshes', _relative_path)
    cfg.blender_path = 'blender'
    cfg.motion_video_read_fps = 30

    return cfg
