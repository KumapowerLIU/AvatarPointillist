CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 -m lam.launch train.lam --config configs/lam-20k-nersemble.yaml
