from pathlib import Path
from safetensors.torch import load_file
from avatarpointillist.gstransformer.models import model_dict


def build_model(cfg, exp_type="lam"):
    if not cfg.get("model_name"):
        raise ValueError("model_name is required")
    checkpoint_path = Path(cfg.model_name) / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # The complete checkpoint contains the DINOv2 encoder. Avoid an unnecessary network download.
    model_args = dict(cfg.model)
    model_args["encoder_pretrained"] = False
    model = model_dict[exp_type](**model_args)
    model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    return model.eval()
