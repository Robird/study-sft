# study-sft: 后训练入门实验台

这个仓库的目标不是一次性把 RL Agent 范式做完，而是先建立一个能复现实验的后训练最小闭环：看数据、格式化、LoRA/QLoRA SFT、加载 LoRA 推理、对比不同文本协议。

## 当前实验目标

长期方向是把被动 Chat 的 `{System/User/Assistant}` 语义，逐步替换成环境中连续行动的 `{Belief/Observation/Reasoning/Action}` 语义。第一阶段先用成熟指令数据集做 SFT，把训练管线、模板、loss 和推理验证摸清楚。

本仓库提供三种 `prompt_mode`：

- `chatml`：标准 ChatML/Qwen 风格，system 在最前面。
- `late_system`：把 system 放到最后一个 assistant 目标输出之前，用来做“局部性更强的动态上下文”A/B 实验。
- `bora`：把输入侧改成 `belief` 和 `observation`，输出侧生成 `Reasoning` 与 `Action`，作为非 Chat 语义的第一块脚手架。

## 快速开始

先离线预览 tiny 数据，确认同一条样本在三种协议下长什么样：

```bash
bash scripts/preview_tiny.sh
```

跑一个 5 step 冒烟训练，只验证环境、显存、保存和 LoRA 加载路径：

```bash
bash scripts/train_tiny_smoke.sh
bash scripts/infer_smoke_adapter.sh
```

正式一点的入门 SFT，对 `yahma/alpaca-cleaned` 跑 500 step：

```bash
bash scripts/train_alpaca_chatml.sh
```

当前本机 `bitsandbytes` 会报 `libnvJitLink.so.13` 缺失，所以脚本默认使用 bf16 LoRA + `adamw_torch`，不走 4bit 量化和 8bit optimizer。修好 CUDA 13 runtime 的 `LD_LIBRARY_PATH` 后，可以给训练脚本加 `--load_in_4bit true --optim adamw_8bit` 切回 QLoRA 路径。

做模板局部性对照实验时，保持数据、seed、学习率、step 数不变，只换 prompt mode：

```bash
bash scripts/train_alpaca_chatml.sh
bash scripts/train_alpaca_late_system.sh
bash scripts/train_alpaca_bora.sh
```

## 你需要学清楚的核心点

SFT 的本质是 next-token prediction，不是“调用 API 学会聊天”。所谓 Chat SFT，只是把样本序列化成某种文本协议，然后继续做语言模型训练。

最重要的四件事：

- 数据协议：模型真正看到的是 token 序列，`system/user/assistant` 只是被模板写进文本后的模式。
- loss 范围：入门脚本先训练整段文本，之后可以升级为只对 assistant/action 部分算 loss。
- 分布迁移：模板改得越激进，越需要推理时也使用同样模板。
- 对照实验：一次只改一个变量，否则你不知道提升来自数据、模板、步数还是学习率。

## 建议里程碑

1. 跑通 `tiny` 冒烟：理解样本如何变成 token、LoRA adapter 保存在哪里、怎么加载推理。
2. 跑 `chatml` Alpaca SFT：得到一个会按 ChatML 做基础指令响应的 adapter。
3. 跑 `late_system` 对照：用包含冲突规则的评测提示检查“靠近输出的 system”是否更容易生效。
4. 跑 `bora` 对照：确认模型是否能稳定输出 `Reasoning/Action` 外壳。
5. 构造真正的 BO-RA 数据：不要再从 Alpaca 硬转，而是记录环境状态、观察、内部信念更新、动作和结果。
6. 进入 RL：在环境里 rollout，用 reward 或 preference 优化连续行动，而不是只优化单轮回答格式。

## 文件地图

- `src/study_sft/formats.py`：三种文本协议的核心格式化逻辑。
- `src/preview_data.py`：预览数据行如何变成训练文本。
- `src/train_sft.py`：Unsloth + TRL + LoRA/QLoRA 训练入口。
- `src/infer_lora.py`：加载 LoRA adapter 做快速生成验证。
- `examples/tiny_alpaca.jsonl`：离线冒烟样本。
- `scripts/*.sh`：常用实验命令。

## 下一步值得补的东西

当前版本故意保持简单，后续最值得补两项：

- assistant/action-only loss mask：避免模型在训练时学习复述输入协议。
- 小型评测集：固定 20 到 50 条提示，专门测格式遵循、system 局部性、BO-RA 连续性和动作可执行性。

## 已验证结果

- `bash scripts/preview_tiny.sh`：通过，能预览 `chatml`、`late_system`、`bora` 三种文本协议。
- `bash scripts/train_tiny_smoke.sh`：通过，5 step tiny LoRA 训练完成，`train_loss` 约 4.40。
- `bash scripts/infer_smoke_adapter.sh`：通过，能加载 smoke adapter 并生成回答。