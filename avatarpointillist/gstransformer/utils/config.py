import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, unknown = parser.parse_known_args()

    assert args.config is not None, "config is required"
    cfg = OmegaConf.load(args.config)
    if 'base_config' in cfg:
        base_path = Path(cfg.base_config)
        if not base_path.is_absolute():
            base_path = Path(args.config).resolve().parent / base_path
        base_cfg = OmegaConf.load(base_path)
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

    dump_root = cfg.get('output_dir', './outputs/decoder')
    cfg.save_tmp_dump = os.path.join(dump_root, 'save_tmp', _relative_path)
    cfg.image_dump = os.path.join(dump_root, 'images', _relative_path)
    cfg.video_dump = os.path.join(dump_root, 'videos', _relative_path)
    cfg.mesh_dump = os.path.join(dump_root, 'meshes', _relative_path)
    cfg.blender_path = 'blender'
    cfg.motion_video_read_fps = cfg.get('motion_video_read_fps', 30)

    return cfg
