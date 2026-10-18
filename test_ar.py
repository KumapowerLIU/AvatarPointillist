"""Generate Gaussian geometry from portrait images and tracked FLAME shape parameters."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import Dinov2Model, AutoImageProcessor
from tqdm import tqdm

from point_encoder.miche.encode import load_model
from avatarpointillist.argeometry.Transformer.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p
from avatarpointillist.argeometry.Transformer.decoder import Decoder
from avatarpointillist.argeometry.Transformer.transformerwarper import TransformerWrapper
from avatarpointillist.argeometry.dataset.datasetgpt_test import DataHeadARGPTTest
from avatarpointillist.argeometry.dataset.data_utils import save_ply
from avatarpointillist.gstransformer.models.rendering.flame_model.flame_GSA import FlameHead


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_test_model(args, device):
    decoder = Decoder(dim=1024, depth=24, heads=16, cross_attend=True,
                      rotary_pos_emb=True, attn_flash=True, use_rmsnorm=True)
    net = TransformerWrapper(num_tokens=args.num_faces + args.n_discrete_size + 3,
                             max_seq_len=args.window_size, attn_layers=decoder)
    model = AutoregressiveWrapper(net, pad_value=args.num_faces + args.n_discrete_size + 2)
    # Original training archives contain Namespace/OmegaConf metadata. Trusted files only.
    checkpoint = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    if args.compile:
        model = torch.compile(model)
    dino = Dinov2Model.from_pretrained(args.dino_path).requires_grad_(False).to(device).eval()
    encoder = load_model(args.point_encoder_ckpt_path).requires_grad_(False).to(device).eval()
    return model, dino, encoder, AutoImageProcessor.from_pretrained(args.dino_path)


def process_condition_data(batch_data, dino_model, point_encoder, device):
    face = dino_model(batch_data["dino_imgs"].to(device)).last_hidden_state
    points = point_encoder.encode_latents(batch_data["pc_normals"].to(device))
    return face, points, point_encoder.to_shape_latents(points[:, 1:])


def get_type_mask(tokens_to_generate, device, xyz_size=1024, face_nums=10144, group_size=4):
    """Four repeating rows avoid allocating a sequence-length by vocabulary tensor."""
    if group_size != 4:
        raise ValueError("Geometry uses four tokens per point")
    mask = torch.full((4, xyz_size + face_nums + 3), float("-inf"), device=device)
    mask[:3, :xyz_size] = 0
    mask[3, xyz_size:xyz_size + face_nums] = 0
    # End only between complete points, never inside an xyz/binding tuple.
    mask[0, xyz_size + face_nums + 1] = 0
    return mask


def from_index_to_point(generated_ids, coord_bins=1024, bound_lower=None,
                        bound_upper=None, output_path=None, num_faces=10144):
    """Decode BOS + complete xyz/binding tuples; stop at the first EOS/PAD."""
    bos, eos, pad = (coord_bins + num_faces + i for i in range(3))
    rows = generated_ids.detach().cpu().numpy()
    xyz_list, binding_list = [], []
    for batch_idx, row in enumerate(rows):
        if len(row) < 4 or not np.all(row[:4] == bos):
            raise ValueError("Expected four BOS tokens")
        row = row[4:]
        stop = np.flatnonzero((row == eos) | (row == pad))
        if len(stop):
            row = row[:stop[0]]
        if len(row) % 4:
            raise ValueError("Generated geometry ends inside a point tuple")
        groups = row.reshape(-1, 4)
        valid = ((groups[:, :3] >= 0) & (groups[:, :3] < coord_bins)).all(axis=1)
        valid &= (groups[:, 3] >= coord_bins) & (groups[:, 3] < coord_bins + num_faces)
        if not valid.all():
            raise ValueError("Invalid coordinate or FLAME face token")
        xyz = groups[:, :3].astype(np.float32) / (coord_bins - 1)
        if bound_lower is not None and bound_upper is not None:
            xyz = xyz * (np.asarray(bound_upper) - np.asarray(bound_lower)) + np.asarray(bound_lower)
        binding = groups[:, 3].astype(np.int64) - coord_bins
        xyz_list.append(xyz)
        binding_list.append(binding)
        if output_path is not None:
            save_ply(f"{output_path}_batch_{batch_idx}.ply", xyz)
    return xyz_list, binding_list


def collate_fn(data):
    return {
        "sample_ids": [item["sample_id"] for item in data],
        "dino_imgs": torch.stack([item["dino_img"] for item in data]),
        "sequences": torch.stack([torch.as_tensor(item["sequence"], dtype=torch.long) for item in data]),
        "pc_normals": torch.stack([torch.as_tensor(item["pc_normal"], dtype=torch.float32) for item in data]),
    }


def filter_sampling(logits, k, thres):
    return top_p(top_k(logits, k=k), thres=thres)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_dir', default='configs/ar_inference.yaml')
    parser.add_argument('--pretrained_weights', default='pretrain_model/checkpoint_ar.pth')
    parser.add_argument('--output_dir', default='outputs/ar')
    parser.add_argument('--max_length', type=int, default=120000, help='Maximum new tokens; multiple of four')
    parser.add_argument('--window_size', type=int, default=9600)
    parser.add_argument('--file_path_test', default=None)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--topp', type=float, default=0.95)
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compile', action='store_true')
    cli = parser.parse_args()
    if cli.max_length <= 0 or cli.max_length % 4 or cli.window_size < 4:
        parser.error('max_length must be positive and divisible by four; window_size must be >= 4')
    if cli.topk <= 0 or not 0 < cli.topp <= 1 or cli.temp < 0:
        parser.error('Require topk > 0, 0 < topp <= 1, and temp >= 0')
    cfg = OmegaConf.merge(OmegaConf.load(cli.config_dir), vars(cli))
    if cli.file_path_test is not None:
        cfg.file_path_train = cli.file_path_test
    args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))
    if not torch.cuda.is_available():
        raise RuntimeError('This release requires a CUDA GPU and its rendering dependencies')
    device = torch.device('cuda')
    set_seed(args.seed)
    model, dino, encoder, _ = load_test_model(args, device)
    root = Path(args.flame_base_path)
    flame = FlameHead(300, 100, flame_model_path=str(root / 'flame2023.pkl'),
                      flame_lmk_embedding_path=str(root / 'landmark_embedding_with_eyes.npy'),
                      flame_template_mesh_path=str(root / 'head_template_mesh.obj'),
                      flame_mask_path=str(root / 'FLAME_masks.pkl')).to(device).eval()
    dataset = DataHeadARGPTTest(args, device=device, test=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    type_mask = get_type_mask(args.max_length, device, args.n_discrete_size, args.num_faces)
    with torch.inference_mode():
        for batch in tqdm(loader, desc='Generating subjects'):
            sample_id = batch['sample_ids'][0]
            if sample_id in {'.', '..'} or '/' in sample_id or '\\' in sample_id:
                raise ValueError(f'Unsafe sample ID: {sample_id}')
            output = Path(args.output_dir) / sample_id
            output.mkdir(parents=True, exist_ok=True)
            face, points, latents = process_condition_data(batch, dino, encoder, device)
            ids, embeddings = model.generate(
                batch['sequences'].to(device), args.max_length,
                eos_token=args.n_discrete_size + args.num_faces + 1,
                temperature=args.temp, type2mask=type_mask,
                filter_logits_fn=filter_sampling, filter_kwargs={'k': args.topk, 'thres': args.topp},
                face_features=face, pc_feature=points, shape_latents=latents)
            xyz_list, bindings = from_index_to_point(ids, args.n_discrete_size,
                np.array(args.global_min, dtype=np.float32), np.array(args.global_max, dtype=np.float32),
                str(output / 'points'), args.num_faces)
            xyz, binding = xyz_list[0], bindings[0]
            if not len(xyz):
                raise RuntimeError(f'Model generated EOS before any points for {sample_id}')
            # Decoder consumes predictor features starting at BOS position 3: exactly 4*N rows.
            embeddings = embeddings[:, :3 + 4 * len(xyz)]
            if embeddings.shape[1] != 3 + 4 * len(xyz):
                raise RuntimeError('Generated point/feature counts disagree')
            stem = f'topk_{args.topk}_topp_{args.topp}_temp_{args.temp}'
            np.save(output / f'{stem}.npy', ids.cpu().numpy())
            np.save(output / f'{stem}_embed.npy', embeddings.cpu().numpy())
            np.save(output / 'new_point_data.npy', {'xyz': xyz, 'binding': binding})
            lbs, shapedirs = flame.calculate_interpolated_properties(
                my_v_cano=torch.as_tensor(xyz, device=device, dtype=torch.float32),
                face_indices_for_my_points=torch.as_tensor(binding, device=device, dtype=torch.long))
            np.savez(output / 'gs_interp.npz', my_lbs_weights=lbs.cpu().numpy(), my_shapedirs_up=shapedirs.cpu().numpy())
            (output / 'generation.json').write_text(json.dumps({
                'sample_id': sample_id, 'num_points': len(xyz), 'seed': args.seed,
                'ar_feature_file': f'{stem}_embed.npy', 'prompt_tokens': 4,
                'stopped_at_eos': bool((ids == args.n_discrete_size + args.num_faces + 1).any()),
            }, indent=2) + '\n')


if __name__ == '__main__':
    main()
