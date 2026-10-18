"""Download the owner-controlled model archive without loading any checkpoint."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='KumaPower/AvatarPointillist')
    parser.add_argument('--revision', default='main', help='Prefer a commit SHA for reproducible downloads')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--include-training-state', action='store_true')
    args = parser.parse_args()
    patterns = ['pretrain_model/checkpoint_ar.pth', 'pretrain_model/shapevae-256.ckpt',
                'pretrain_model/dinov2-base/*.json', 'pretrain_model/dinov2-base/model.safetensors',
                'pretrain_model/gaussian_decoder/model.safetensors', 'toy_data/flame/**']
    if args.include_training_state:
        patterns.append('pretrain_model/gaussian_decoder/*')
    snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=args.output_dir,
                      allow_patterns=patterns)


if __name__ == '__main__':
    main()
