import os
import torch

from typing import Literal
from safetensors.torch import load_file

from gstransformer.models import model_dict


def build_model(cfg, exp_type: Literal['lam'] = 'lam') -> torch.nn.Module:
    # hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
    # model = hf_model_cls.from_pretrained(cfg.model_name)

    model_class = model_dict[exp_type]
    print(f"model class: {model_class}")
    model = model_class(**cfg.model)

    assert cfg.model_name is not None, 'model_name is required'
    resume = os.path.join(cfg.model_name, "model.safetensors")
    print("="*150)
    print("loading pretrained weight from:", resume)
    if resume.endswith('safetensors'):
        ckpt = load_file(resume, device='cpu')
    else:
        ckpt = torch.load(resume, map_location='cpu')
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"[WARNING] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
        else:
            print(f"[WARNING] unexpected param {k}: {v.shape}")
    print("finish loading pretrained weight from:", resume)
    print("="*150)
    return model
