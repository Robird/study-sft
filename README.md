# study-sft: 后训练入门实验台

这个仓库的目标不是一次性把 RL Agent 范式做完，而是先建立一个能复现实验的后训练最小闭环：看数据、归一化成 agentic context、LoRA/QLoRA SFT、加载 LoRA 推理。

## 文档入口

不同文件现在各自承担不同职责，避免一事多写：

- `README.md`：仓库入口、快速开始、实验地图。
- `docs/agentic-context-format-design.md`：agentic context 的层次、边界与当前 `v0` 收敛方向。
- `examples/xml-authoring/01-code-review-and-patch.xml`：XML authoring 的 worked example。
- `examples/agentic-ml/01-ask-and-answer.txt`：最早的概念手稿，保留为历史附录。

如果要确认“当前规范到底是什么”，以 `docs/agentic-context-format-design.md` 为准。

## 当前实验目标

当前阶段已经从三套 prompt baseline 切到单一协议：训练、预览和推理都统一使用 agentic-context。通用指令数据会先映射成共享的 agentic context，再由安全 encoder 产出 `input_ids + loss_mask/labels`。

## 快速开始

先离线预览 tiny 数据，确认样本如何被归一化成 agentic context：

```bash
bash scripts/preview_tiny.sh
```

需要看 token 文本和 debug spans 时，再显式打开重模式：

```bash
python src/preview_data.py \
  --dataset_path examples/tiny_alpaca.jsonl \
  --dataset_format alpaca \
  --show_token_text \
  --show_spans
```

跑一个 5 step 冒烟训练，只验证环境、显存、保存和 LoRA 加载路径：

```bash
bash scripts/train_tiny_smoke.sh
bash scripts/infer_smoke_adapter.sh
```

正式一点的入门 SFT：

```bash
bash scripts/train_alpaca_agentic.sh
```

当前本机 `bitsandbytes` 会报 `libnvJitLink.so.13` 缺失，所以脚本默认使用 bf16 LoRA + `adamw_torch`，不走 4bit 量化和 8bit optimizer。修好 CUDA 13 runtime 的 `LD_LIBRARY_PATH` 后，可以给训练脚本加 `--load_in_4bit true --optim adamw_8bit` 切回 QLoRA 路径。

## 你需要学清楚的核心点

SFT 的本质是 next-token prediction，不是“调用 API 学会聊天”。现在仓库里的核心不是对比 prompt 文本模板，而是把训练样本先归一化成结构化 agentic context，再显式控制哪些 token 参与 loss。

最重要的四件事：

- 数据协议：训练主输入已经不是 `text prompt`，而是共享的 agentic context 表示再编码成 `input_ids/labels`。
- loss 范围：message-level `loss` 会通过 `loss_mask -> labels(-100)` 落到真正的训练目标上。
- 分布迁移：训练和推理都必须走同一套 agentic-context 协议，不能再混用旧 chat prompt。
- 复杂度控制：第一版先禁用 packing / streaming / raw text 数据；多轮 `messages/sharegpt` 会展开成多个“前缀到目标 assistant”为止的单目标训练样本。

## 建议里程碑

1. 跑通 `tiny` 冒烟：理解样本如何变成 agentic context、token、labels，以及 LoRA adapter 保存在哪里。
2. 跑 `alpaca` agentic SFT：确认 `{belief, observation, me}` 角色能稳定收敛到基础回复能力。
3. 补 generation-prefix 和停止条件的更细粒度控制，让推理端更贴近真实 agent loop。
4. 构造真正的 agent-native 数据：不要再从 Alpaca 硬转，而是记录环境状态、观察、内部信念更新、动作和结果。
5. 进入 RL：在环境里 rollout，用 reward 或 preference 优化连续行动，而不是只优化单轮回答格式。

## 文件地图

- `src/study_sft/agentic_context.py`：typed agentic context 到 encoded protocol 的安全编码、validation，以及 generation payload prefix 入口。
- `src/study_sft/samples.py`：把 `alpaca/messages/sharegpt` 归一化成共享 sample，并为多轮对话展开训练用单目标样本。
- `src/study_sft/training_data.py`：训练侧特征编码、loss label 构造，以及按监督消息后缀截断；数据集特征只保留 `input_ids + labels`。
- `src/preview_data.py`：默认预览 context JSON；需要时可额外打印 token 文本和 debug spans。
- `src/train_sft.py`：Unsloth + LoRA/QLoRA + pretokenized `Trainer` 训练入口。
- `src/infer_lora.py`：加载 LoRA adapter，用 agentic generation payload prefix 做快速生成验证。
- `examples/tiny_alpaca.jsonl`：离线冒烟样本。
- `scripts/*.sh`：常用实验命令。

## 下一步值得补的东西

当前版本故意保持简单，后续最值得补几项：

- generation output 的最小 parser / 后处理：把 `me` 输出里的结构边界和 payload 恢复成更易用的结果。
- 小型评测集：固定 20 到 50 条提示，专门测角色遵循、结构闭合、agentic 回复一致性和动作可执行性。
- 真正 agent-native 的训练数据，而不只是把 chat 数据映射成新角色。

## 已验证结果

- `bash scripts/preview_tiny.sh`：通过，能快速预览 agentic context JSON；加 `--show_token_text --show_spans` 可进入编码诊断模式。
- `src/train_sft.py` 的新 pretokenized agentic 训练链路：已手动跑通 1 step smoke，`train_loss` 约 4.57；需要 encoder 自检时可额外加 `--validate_encoding true`。
- `src/infer_lora.py` 的新 generation-prefix 推理链路：已手动跑通单轮推理；1-step smoke adapter 的输出质量仍很粗糙，属预期现象。
