# AvatarPointillist 静态代码审查

审查日期：2026-10-18（Asia/Shanghai）。源代码来自用户提供的 `cvpr2026_code_review` 目录；修改位于 GitHub 仓库的独立分支，原始目录保持不变。

**范围与边界：** 阅读 AR 训练/生成、数据加载、Transformer wrapper、Gaussian decoder 加载/训练/推理、配置和发布依赖；对全部纳入分支的 Python 做 AST 语法分析，对 JSON/YAML 做解析，对主要 Python 做未定义名称静态检查。读取 safetensors JSON 头和 PyTorch ZIP/pickle 指令元数据，但没有反序列化 checkpoint。没有导入项目、运行训练/推理、构建 CUDA 扩展或做数值/图像效果测试。以下“已修复”指代码修改和静态检查，不代表已通过运行验证。

## 已修复的高影响问题

| 等级 | 原问题与影响 | 修改位置 / 处理 |
|---|---|---|
| P1 | Gaussian decoder 配置关闭 binding conditioning，但权重包含 `index_embeddings` 和 `[1024, 3072]` 的 `fusion_layer.weight`。代码原先构建 2048 输入的融合层，又静默跳过形状不匹配参数，会留下随机权重。 | `avatarpointillist/configs/gaussian-decoder-*.yaml` 开启 binding；`gstransformer/utils/build_model.py` 严格加载全部权重，禁止静默漏载。保留 checkpoint 的 101444 项 binding embedding 形状，未擅自改为 10144。 |
| P1 | 推理没有显式 `eval()`，模型仍可能处于训练模式。 | decoder builder 和 inferencer 显式进入 eval；已有完整 encoder 权重时取消初始化阶段的额外下载。 |
| P1 | AR 推理从真实 Gaussian PLY 读取并拼入前 4000 个 token，会使用目标几何信息，不能作为纯条件生成结果。 | `test_ar.py` 和 `datasetgpt_test.py` 只用四个 BOS；测试输入仅需图像及跟踪 shape 参数。没有声称这等同于无需 FLAME 跟踪的原始单图端到端系统。 |
| P1 | 测试循环无条件在第一批后退出；简单移除 break 又会覆盖同名输出。 | 全部样本按 ID 写入独立目录，并校验目录名。 |
| P1 | 未把 EOS 传给 generate；点云转换拿 `-1` 当结束标记，且不检查 binding 范围；结束 token 可落在点的中间。 | 按实际词表定义 EOS/PAD；仅允许完整点之间结束；验证每个 xyz/binding token。 |
| P1 | 用 `120000 × 11171` 的 float32 类型 mask，单这一项约 5.36 GB；循环反复拼接 embedding 会产生大量复制。 | 改成四行循环类型 mask，并累积 feature chunks 后统一拼接。 |
| P1 | generate 在不传类型 mask 时使用未初始化 `seq_id`，关闭长度限制时 `x` 未赋值；关闭 cache 时还会重复保存整段 embedding。 | 每次循环初始化索引和输入；首轮后只取最后一个预测状态；恢复训练模式使用 finally。 |
| P1 | 两阶段间 feature/point 关系在 EOS 后可能偏移；旧 loader 只用整除判断数量，允许多余行进入 reshape。 | 明确第 3 个 BOS 位置起的预测特征为每点四行；生成时裁齐，训练/推理加载时核对精确 `4*N` 长度。 |
| P1 | AR 训练对不同长度序列直接 stack，短序列批次会报错；CUDA 数据集默认用多进程 worker，会触发 CUDA fork 问题。 | `train_ar.py` 使用 pad_sequence 和 -100 忽略标签；固定该 CUDA 数据集的 workers 为 0。 |
| P1 | AR 的学习率总步数在分布式数据切分前估计且向下取整；调度异常被吞掉；加载失败会悄悄从头训练。 | 按准备后的每进程 loader 和向上取整计算更新数；调度只随成功 optimizer 更新前进；加载/调度错误明确抛出。 |
| P1 | 非有限 loss 仅某个 rank continue，其他 rank 可能卡在梯度同步。 | 两条训练路径先跨 rank 汇总有限性，再统一中止。 |
| P1 | decoder 训练保存了未经过 Accelerator 的 `real_train_loader` 别名，实际循环继续使用未切分的数据。 | `base_trainer.prepare_everything()` 更新两个 real-loader 别名；恢复批次跳过和 epoch sampler 设置。 |
| P1 | decoder 中同一 pixel loss 被累加两次，实际权重是配置的两倍。 | 删除重复累计；保留一次配置权重。历史训练结果不能因此宣称已复现。 |
| P1 | `runners/__init__.py` 没有注册训练 runner，公开训练入口找不到 `train.lam`；包导入残留 lam/gstransformer 私有路径。 | 启动时按训练 runner 懒加载注册；统一项目包导入并移除 launcher 的服务器 sys.path。 |
| P1 | 导入 `data_prepare.py` 就加载不存在的 ONNX 模型，即使 RGBA 或不需要抠图也无法启动。 | 改为需要 RGB 抠图时才初始化；支持环境变量路径，缺失时给出明确错误。 |

## 其他已处理问题

- `--window_size` 原本没有影响推理模型的实际 window；现在直接用于构建模型。`--topp` 原本只参与文件名，现在进入采样过滤。
- torch.load 的原始训练 archive 包含 Namespace/OmegaConf 对象，显式使用可信本地文件的 `weights_only=False`；没有在审查过程中反序列化。推理改用 CPU 读取 checkpoint，随后释放 checkpoint 引用。
- 原始配置中的 `Global min/max` 不是代码所需的 `global_min/max`，并且数值列表缺逗号。已改正。
- 训练和测试 manifest 分别要求 singleton-list 与 mapping，已统一兼容两种格式。
- 条件 dropout mask 多出一个维度，且硬绑默认 CUDA；现在按特征的设备和三维形状处理。
- 训练检查点频率有 `+1` 偏移，训练末尾没有最终保存。已修复，并新增 batch/global-step/scaler 元数据。
- decoder `laoder` 拼写错误、YAML `None` 字符串、persistent_workers 随 pin_memory 错误开关等已修复。
- decoder 非 Nersemble / `vis_rgb=false` 路径使用未定义视频列表。现在按条件创建并保存，尊重 export_video。
- decoder 的 AR 特征文件名不再总是硬编码，优先读取 `generation.json`。
- `enlarge_ratio=[1.0, 1,0]` 三元素拼写错误以及硬编码 frames_per_loop 已修复。
- 多处缺失导入（Optional、ModuleList、Decoder、AttentionPool、randrange、nullcontext、autocast、Meshes、Image、AutoencoderKL 等）、错误 `__all__`、工具函数返回未赋值变量已修复。
- 批处理辅助脚本原先只调度前若干 JSON 后宣称成功；现在配置数量超出槽位会明确报错，且 shell 参数使用安全引用。
- 根目录新增候选依赖列表、模型下载脚本、GLM/光栅器构建脚本、样例配置、安装说明和来源说明。原包中的二进制扩展/构建缓存、旧 Git 历史、私人实验 job manifests、scratch debug 脚本和未使用且标明 proprietary 的 ray marcher 未放入公开分支。具体清单见 `release_exclusions.json`。

## 仍需运行或作者确认的事项

1. **效果与论文复现未验证。** 需要作者在 CUDA 环境验证 checkpoint strict-load、生成/EOS/滑窗 KV cache、动画坐标系、渲染效果和多卡训练。此审查不能证明论文指标，也不能证明全部代码没有其他问题。
2. **环境不是锁定环境。** 根依赖由静态 imports 推导；custom Transformer 使用 x-transformers 内部接口，仍需在真实环境确认兼容版本。原包缺 GLM 和 simple-knn 源码；已提供 GLM 获取步骤，Gaussian fitting 仍是独立依赖流程。
3. **缺失模型。** 包中没有 RGB foreground ONNX 模型、自动 FLAME 跟踪模型、训练 LPIPS/AlexNet 下载权重。它们无法从本地包上传。原先三张示例为 RGB；示例 decoder 显式跳过抠图，仅作为输入格式示例，实际原图应提供模型/alpha/mask。
4. **不支持任意 batch 的变长 Gaussian。** decoder 默认 batch size 1；多样本变长点数需要经验证的 padding/mask 或分桶方案。AR 已处理变长 token 批次。
5. **旧 checkpoint 的精确恢复信息不足。** 新保存的 AR checkpoint 记录 batch/global-step/scaler，但完整 RNG/增强状态没有逐 rank 持久化，旧文件也没有全部恢复信息；不承诺 bitwise 相同恢复。断点改变训练总步数时 OneCycle schedule 还需单独设计迁移。
6. **算法相关超参数不能从权重恢复。** decoder offset 上限等非 tensor 参数仍保留测试配置值；需要作者确认与 checkpoint 实验一致。修复重复 pixel loss 会改变后续训练目标。
7. **附带的可选旧代码不等于完整发布接口。** standalone Michelangelo CLIP/text/image generation（含缺失 clip 依赖）、scratch trainer、旧 tracking/viewer 和额外 dataset 类型并未补齐。仅文档指定的 AR/decoder 路径作为本分支入口；历史辅助脚本仍可能含旧路径。
8. **发布许可未统一。** 存在 Apache/MIT、NVIDIA/INRIA、GaussianAvatars 非商业、FLAME/SMPL 系列等不同来源声明；没有自动为整个包设置一个新许可。decoder safetensors 内含 FLAME 模板/shape 等 buffers，因此模型先归档到私有 HF 仓库，公开再分发条件需另行明确。代码中的原有声明保留；详见 `THIRD_PARTY.md`。

## 模型证据与交付

- AR 原始 checkpoint：5,160,441,282 bytes。
- Gaussian decoder：3,254,835,588 bytes，header 含 1,137 个 tensor（另有 metadata）；fusion `[1024,3072]`、binding embedding `[101444,1024]`。
- Shape VAE：3,934,164,973 bytes。
- DINOv2：safetensors 与 PyTorch bin 两种原始格式均保留；decoder optimizer、random state 和所有随包 FLAME 模型资产也归档。
- 总计 34 个模型/配置/资产文件，16,143,936,685 bytes，校验值见根目录 `models_manifest.json`。HF 仓库为 `KumaPower/AvatarPointillist`（私有）。上传完成状态和远端校验以 `MODEL_UPLOAD_STATUS.json` 为准。

静态检查输出概要见 `static_checks.json`。pyflakes 对 jaxtyping 的维度字符串存在 forward-annotation 误报，不能把它当作 Python AST 语法错误；unused imports 和历史可选分支告警也不等于已全部消除。
