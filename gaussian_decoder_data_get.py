#!/usr/bin/env python3
"""
按配置目录把每个 sample JSON 分配到多卡并在 tmux 中启动进程。

示例用法：
  python3 scripts/run_tmux_jobs.py \
    --configs-dir ./configs/cluster_processing_ar \
    --script-path ./test_ar_gpt.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --jobs-per-gpu 3 \
    --extra-args "--temp 0 --topk 1" \
    --out-dir ./test_outputs_ar

说明：
- 每个 JSON 文件会单独启动一个 tmux session，运行命令：
    CUDA_VISIBLE_DEVICES=<gpu> python3 <script-path> --file_path_test <json> <extra-args> --output_dir <out-dir>/<session>
- 若 tmux 未安装或为 dry-run，只会打印将要执行的命令。
"""
import argparse
import subprocess
from pathlib import Path
import shutil
import sys
import os
def make_cmd(gpu, script_path, cfg_path, extra_args, out_dir, session_name):
    # ensure file_path_test and output_dir present
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python3 {script_path} --file_path_test {cfg_path} {extra_args} "
        f"--output_dir {out_dir}/{session_name} "
        f"> {out_dir}/{session_name}.log 2>&1"
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs-dir", default='./configs/cluster_processing_ar2', help="包含单个 sample json 的目录（cluster_processing_ar）")
    p.add_argument("--script-path", default='test_ar_gpt.py', help="要运行的 python 脚本路径（例如 test_ar_gpt.py）")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="逗号分隔的 GPU id 列表，例如 0,1,2,3,4,5,6,7")
    p.add_argument("--jobs-per-gpu", type=int, default=1, help="每张卡同时运行的程序数（每个程序一个 tmux 会话）")
    p.add_argument("--extra-args", default="--temp 0 --topk 1", help="传给脚本的额外参数（例如 \"--temp 0 --topk 1\"）")
    p.add_argument("--out-dir", default="/robby/share/Avatar/hongyul/ar_out", help="公共输出目录，脚本会为每个会话创建子目录并写日志")
    p.add_argument("--session-prefix", default="arjob", help="tmux 会话名前缀")
    p.add_argument("--dry-run", action="store_true", help="仅打印要执行的命令，不启动 tmux 会话")
    args = p.parse_args()

    configs_dir = Path(args.configs_dir)
    if not configs_dir.exists() or not configs_dir.is_dir():
        print("configs-dir 不存在或不是目录：", configs_dir)
        sys.exit(2)

    script_path = Path(args.script_path)
    if not script_path.exists():
        print("script-path 不存在：", script_path)
        sys.exit(3)

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if len(gpu_list) == 0:
        print("未指定 GPU 列表")
        sys.exit(4)

    json_files = sorted([p for p in configs_dir.iterdir() if p.suffix.lower() == ".json"])
    # json_files = json_files[40:] # 2是40开始
    # json_files = json_files[80:] # 3是80开始
    # json_files = json_files[120:] # 4是120开始
    # json_files = json_files[160:] # 1第二轮是160开始
    # json_files = json_files[200:] # 2第二轮是200开始
    # json_files = json_files[240:] # 3第二轮是240开始
    # json_files = json_files[280:] # 4第二轮是280开始
    # json_files = json_files[320:] # 2第三轮是320开始

    if len(json_files) == 0:
        print("configs-dir 中没有 json 文件：", configs_dir)
        sys.exit(5)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_tmux = shutil.which("tmux") is not None
    if not use_tmux:
        print("警告：系统找不到 tmux，可使用 --dry-run 或手动安装 tmux。将以 shell 方式打印命令。")

    cmds = []
    # 按 GPU 轮询分配：每张卡启动 jobs_per_gpu 个任务，直到用完 json_files
    it = iter(json_files)
    sess_idx = 0
    for gpu in gpu_list:
        for j in range(args.jobs_per_gpu):
            try:
                cfg = next(it)
            except StopIteration:
                break
            file_base_name = os.path.splitext(os.path.basename(cfg))[0]
            # print(file_base_name)
            session_name = f"{args.session_prefix}_g{gpu}_j{j}_{file_base_name}"
            cmd = make_cmd(gpu, script_path, cfg, args.extra_args, out_dir, session_name)
            cmds.append((session_name, cmd))
            sess_idx += 1
        # 如果 json 全分配完，退出最外层
        if sess_idx >= len(json_files):
            break

    # 若还有未分配的 json（gpu-list 不足以容纳所有 jobs），继续循环分配
    # while True:
    #     try:
    #         cfg = next(it)
    #     except StopIteration:
    #         break
    #     # 继续按 GPU 轮转分配
    #     gpu = gpu_list[(sess_idx // args.jobs_per_gpu) % len(gpu_list)]
    #     j = sess_idx % args.jobs_per_gpu
    #     session_name = f"{args.session_prefix}_g{gpu}_j{j}_{sess_idx}"
    #     cmd = make_cmd(gpu, script_path, cfg, args.extra_args, out_dir, session_name)
    #     cmds.append((session_name, cmd))
    #     sess_idx += 1

    # 启动/打印
    started = []
    for session_name, cmd in cmds:
        if args.dry_run or not use_tmux:
            print("[DRY] tmux session:", session_name)
            print("      cmd:", cmd)
            started.append(session_name)
            continue
        # 新建后台 tmux session 并运行命令
        tmux_cmd = ["tmux", "new-session", "-d", "-s", session_name, cmd]
        try:
            subprocess.run(tmux_cmd, check=True)
            print("Started tmux session:", session_name)
            started.append(session_name)
        except subprocess.CalledProcessError as e:
            print("启动 tmux 失败:", session_name, e)

    print(f"总共准备 {len(cmds)} 个会话，成功启动/打印 {len(started)} 个")
    print("查看 tmux 会话： tmux ls")
    print("附加到某个会话： tmux attach -t <session_name>")

if __name__ == "__main__":
    main()