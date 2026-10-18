1. 单机多卡训练：

```bash
# 使用 torchrun (推荐)
torchrun --nproc_per_node=4 ddp_trainer.py --config config.yaml

# 或者手动启动
python -m torch.distributed.launch --nproc_per_node=4 ddp_trainer.py --config config.yaml
```

2. 多机多卡训练：

```bash
# 节点0 (master)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:29400 ddp_trainer.py

# 节点1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:29400 ddp_trainer.py
```

3. SLURM 集群：
```bash
sbatch slurm_script.sh
```
