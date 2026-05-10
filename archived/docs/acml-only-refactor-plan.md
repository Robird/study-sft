# ACML-Only Refactor Plan

## 状态

面向当前实验项目的重构计划。目标是移除传统对话数据兼容层，聚焦 ACML authoring / parser / adapter / training 这条主线。

本文档既是实施计划，也是后续向 subagent 分发任务时的 task brief 基础。

## 背景与动机

当前仓库已经形成了两条并存的数据入口：

- 传统数据入口：`alpaca / messages / sharegpt -> normalized conversation -> training sample -> AgenticContext`
- ACML 入口：`acml record -> ACML parser / semantic model / adapter -> AgenticContext`

随着 ACML 成为当前实验主线，这种“双入口并存”开始带来额外复杂度：

- `preview_data.py` 同时维护两套主流程。
- `training_data.py` 同时维护两套样本展开逻辑。
- `samples.py` 同时承担传统数据归一化与 ACML bridge。
- CLI、脚本、README、缓存 identity 都还暴露传统格式概念。

如果当前项目明确聚焦 ACML，那么这些兼容层不再是必要能力，而是会稀释主线表达、抬高维护成本。

## 目标

本轮重构的目标是：

1. 当前训练与预览链路只支持 ACML 数据。
2. `record -> AgenticContext -> encoded training features` 成为唯一主链。
3. 删除当前项目内部对 `alpaca / messages / sharegpt` 的训练兼容与 CLI 暴露。
4. 文档、脚本、缓存 identity、测试一起同步收口。

## 非目标

本轮不做：

- 为传统数据保留隐式兼容入口。
- 在当前项目内实现新的传统数据导入器。
- 改动 ACML parser / semantic model / adapter 的核心分层。
- 改动推理侧 `conversation_from_user_text()` 所服务的交互入口，除非实现时发现强耦合问题。

传统数据如果后续还需要，应作为单独导入/导出工具存在，而不是继续耦合在训练主链里。

## 目标形状

重构后，训练与预览链路应统一成：

```text
dataset row
  -> ACML text
  -> ACML parser
  -> ACML syntax model
  -> ACML semantic model
  -> study_sft ACML adapter
  -> AgenticContext
  -> encoder / labels / preview
```

从模块职责上看，应收敛为：

- `src/acml/*`
  负责 ACML 文本、语法模型、语义模型、序列化。
- `src/study_sft/adapters/acml.py`
  负责 ACML 到 `study_sft v0 core IR` 的 bridge / projection。
- `src/study_sft/samples.py`
  负责 ACML dataset row 到 training `AgenticContext` 的最薄投影入口。
- `src/study_sft/training_data.py`
  只负责 `AgenticContext -> input_ids + labels`。
- `src/preview_data.py`
  只负责预览 ACML row 如何成为 training `AgenticContext` 与编码结果。

## 关键设计决定

### 1. `AgenticContext` 作为唯一下游公共桥

下游模块不再关心：

- `dataset_format`
- `NormalizedConversation`
- `TrainingSample`
- assistant-turn expansion

这些都不应再出现在训练与预览主链中。

下游只消费：

- 单条 ACML record
- 或由上游已经投影好的 `AgenticContext`

### 2. `training_data.py` 不再感知数据格式

`training_data.py` 的职责应收敛为：

- 编码一个 training `AgenticContext`
- 根据 label policy 生成 labels
- 做监督后缀截断

因此：

- `iter_training_features_from_record()` 应只通过 ACML path 取到 `AgenticContext`
- `encode_training_sample()` 若仍保留，也应退化为兼容 wrapper，而不是主实现

### 3. `preview_data.py` 不再维护双分支

预览主链应只做：

1. 加载 record
2. 投影成 training `AgenticContext`
3. 打印 context
4. 可选打印 token text / spans

不再需要：

- `dataset_format == "acml"` 的特判分支
- `conversation_from_record()`
- `training_samples_from_conversation()`
- `agentic_context_from_sample()`

### 4. CLI 与脚本应显式承认 ACML-only

训练与预览入口应移除：

- `--dataset_format`
- `--belief_prompt`

并默认认为输入就是 ACML dataset。

如果未来需要转换传统数据，应在仓库外或单独工具脚本中完成，再产出 `.acml` 或带 `acml` 列的数据集。

### 5. 缓存 identity 也应收口

`TrainingEncodingConfig` 与 `training_dataset` cache identity 里，不再保留：

- `dataset_format`
- `belief_prompt`

缓存键只反映 ACML-only 路径真正相关的编码因素。

## 预期移除或收缩的内容

### `src/study_sft/samples.py`

预期移除：

- `DatasetFormat`
- `DATASET_FORMAT_CHOICES`
- `NormalizedRole`
- `NormalizedTurn`
- `NormalizedConversation`
- `TrainingSample`
- `conversation_from_record()`
- `training_samples_from_conversation()`
- `agentic_context_from_conversation()`
- `agentic_context_from_sample()`

预期保留或新增：

- ACML-only 的 `agentic_context_from_record()` / `agentic_contexts_from_record()` 薄入口
- 推理侧若仍使用的 `conversation_from_user_text()` 可暂时保留，但应明确它不属于训练数据主链

### `src/study_sft/training_data.py`

预期收敛：

- 保留 `encode_training_context()`
- `iter_training_features_from_record()` 统一走 ACML context path
- `encode_training_sample()` 若无外部必要依赖，可删除；否则降级为薄 wrapper

### `src/preview_data.py`

预期收敛：

- 只展示 ACML record -> training context
- 提取重复的编码/打印 helper
- 删除传统数据的会话展开展示逻辑

### 训练入口 / 脚本 / README

预期收敛：

- `train_sft.py` 不再接受 `dataset_format` / `belief_prompt`
- README 和脚本不再以 Alpaca 作为主线示例
- 示例与说明改为 `.acml` 或带 `acml` 列的数据集

## 实施顺序

建议按以下顺序推进：

1. 收敛 `samples.py`
2. 收敛 `training_data.py`
3. 收敛 `preview_data.py`
4. 收敛 `cli_args.py`, `train_sft.py`, `training_dataset.py`
5. 更新 README、脚本、测试

这样可以先把核心主链拉直，再处理外围表述。

## 测试策略

至少需要覆盖：

1. ACML record 能投影成 training `AgenticContext`
2. 训练编码仍能正确产生 supervised labels
3. `payload_only` label policy 仍然成立
4. 训练集缓存 identity 在 ACML-only 配置下稳定
5. preview 的 token text / spans 调试输出仍正常

同时应删除或改写那些只验证传统数据兼容逻辑的测试。

## 风险与注意事项

### 1. 推理侧与训练侧不要误删同名概念

`conversation_from_user_text()` 服务于推理交互，不等同于训练数据兼容层。

如果保留它，应在文档和代码注释里明确：

- 它是 inference convenience helper
- 不是 training dataset ingestion path

### 2. 测试需要同步换主入口

当前部分测试直接 patch：

- `conversation_from_record`
- `training_samples_from_conversation`
- `agentic_context_from_sample`

这些测试在 preview / training 收口后，需要改成围绕：

- `agentic_contexts_from_record`
- `agentic_context_from_acml_record`
- `encode_training_context`

### 3. 不要把“聚焦 ACML”误做成“ACML 特殊分支”

目标是让 ACML 成为默认主线，而不是在现有双分支架构里把另一支删掉后留下大量 `acml` 特判残迹。

理想状态是：

- 训练与预览模块天然就是 ACML-only
- 代码里不再大量出现 “if dataset_format == 'acml'”

## 并行实施分片

为了便于多 agent 并行，可按以下分片：

### 分片 A：主链数据与训练

文件：

- `src/study_sft/samples.py`
- `src/study_sft/training_data.py`
- `tests/test_training_data.py`

### 分片 B：预览与 CLI

文件：

- `src/preview_data.py`
- `src/study_sft/cli_args.py`
- `tests/test_agentic_context.py`

### 分片 C：训练入口、缓存、README、脚本

文件：

- `src/train_sft.py`
- `src/study_sft/training_dataset.py`
- `README.md`
- `scripts/*.sh`
- `tests/test_training_runtime.py`

## 完成标准

当以下条件都满足时，可认为本轮重构完成：

1. 当前训练/预览主链只接受 ACML 数据。
2. 传统数据集兼容层已从当前项目主链移除。
3. 文档与脚本不再把 Alpaca / ShareGPT 作为当前主线。
4. 核心测试通过。
5. 代码边界比重构前更清晰，而不是只做表面删除。
