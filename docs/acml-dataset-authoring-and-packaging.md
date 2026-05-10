# ACML Dataset Authoring And Packaging

## 状态

当前训练入口已经收口为 ACML-only。

这意味着：

- 训练链路不再在项目内部兼容 Alpaca / ShareGPT / messages 等传统格式。
- 训练样本必须先整理成 ACML。
- 如果后续仍需要从传统数据导入，应作为单独工具完成，而不是继续耦合在训练主链里。

本文说明：

1. 单条训练样本应如何 authoring
2. 多条样本应如何打包
3. 当前训练入口接受哪些数据载体

## 1. 训练真正消费的是什么

当前训练主链可以概括为：

```text
ACML authoring text
  -> ACML parser
  -> ACML syntax model
  -> ACML semantic model
  -> study_sft ACML adapter
  -> AgenticContext
  -> encoder
  -> input_ids + labels
```

如果只看共享 ACML 层，建议把边界理解为：

- `Document`：syntax model，用于 parser / serializer 的保真读写。
- `SemanticDocument`：lossless semantic envelope，用于跨工具的语义级导入导出。
- `SemanticContext`：lossy entry projection，用于训练 adapter 这类只关心顶层流的下游。

训练时真正需要的是：

- 一条条 ACML document
- 每条 document 能被投影成一个 training `AgenticContext`
- 其中至少有一条 entry 明确参与监督

## 2. 单条样本的最低要求

一条可训练 ACML 样本至少应满足：

- 根元素使用 `<acml version="0"> ... </acml>`
- 每条顶层分段使用 `<acml:entry kind="..."> ... </acml:entry>`
- kind 使用当前项目已稳定的 `belief` / `observation` / `me`
- 至少有一条 entry 带 `loss="true"`

当前训练入口默认按 `explicit` supervision policy 解释 ACML。

也就是说：

- 只有显式 `loss="true"` 的句子会参与监督
- 只有显式 `loss="true"` 的 entry 会参与监督
- 没有任何 `loss="true"` 的 document 会在训练编码阶段报错

最小可训练示例：

```acml
<acml version="0">
<acml:entry kind="belief">You are a helpful, honest, and concise assistant.</acml:entry>
<acml:entry kind="observation">Explain what supervised fine-tuning is in one sentence.</acml:entry>
<acml:entry kind="me" loss="true">Supervised fine-tuning teaches a model from labeled input-output examples.</acml:entry>
</acml>
```

## 3. 推荐 authoring 约定

为了让训练样本更稳定、也更便于后续自动处理，推荐遵守这些约定：

- 一条 ACML document 只表达一个训练样本。
- 推荐每条样本只有一个主要监督目标，即只给一条 `kind="me"` entry 标 `loss="true"`。
- `belief` 放稳定规则、身份、运行模式。
- `observation` 放用户输入、工具结果、材料、文件片段等外部信息。
- `me` 放模型当前这一步应该产出的内容。
- 大段不可信原文、工具输出、文件内容优先放进 `<acml:payload> ... </acml:payload>`。
- 文件保存为 UTF-8 文本，扩展名推荐 `.acml`。

当前训练链路并不强制“每条 document 只能有一个 `loss="true"`”。

但对大多数 SFT 数据集来说，推荐先保持：

- 一个 document
- 一个主目标 `me`
- 一个显式 `loss="true"`

这样最直观，也最容易检查。

## 4. 推荐的打包方式

当前训练入口接受三种常见载体。

### 4.1 单个 `.acml` 文件

适合：

- smoke test
- 手工打磨的单样本
- 极小规模实验

命令示例：

```bash
python src/train_sft.py --dataset_path /path/to/sample.acml
```

此时 loader 会把这个文件包装成一个只含一行 `acml` 列的数据集。

### 4.2 `jsonl` / `json` 文件，带 `acml` 列

这是当前最推荐的批量样本打包方式。

每一行或每一条记录至少应包含：

```json
{"acml": "<acml version=\"0\">...</acml>"}
```

`jsonl` 示例：

```jsonl
{"acml": "<acml version=\"0\"><acml:entry kind=\"observation\">Q1</acml:entry><acml:entry kind=\"me\" loss=\"true\">A1</acml:entry></acml>"}
{"acml": "<acml version=\"0\"><acml:entry kind=\"observation\">Q2</acml:entry><acml:entry kind=\"me\" loss=\"true\">A2</acml:entry></acml>"}
```

命令示例：

```bash
python src/train_sft.py --dataset_path /path/to/train.jsonl
```

注意：

- `acml` 字段必须是字符串
- 需要用标准 JSON 转义换行和双引号
- 不建议手写转义，最好由程序生成

### 4.3 `datasets` 保存的本地目录或 Hub dataset

适合：

- 较大数据集
- 需要标准 split 管理
- 需要复用 `datasets` 生态工具链

要求是：

- 对应 split 中存在 `acml` 列
- `acml` 列每行都是一条 ACML document 字符串

命令示例：

```bash
python src/train_sft.py \
  --dataset_path /path/to/acml_dataset_dir \
  --dataset_split train
```

或者：

```bash
python src/train_sft.py \
  --dataset_name your-org/your-acml-dataset \
  --dataset_split train
```

## 5. 当前不推荐的打包方式

当前不推荐：

- 在训练时再临时从 Alpaca / ShareGPT 映射
- 把多条样本拼进同一个 `.acml` document 再靠自定义切分器拆
- 在 attribute 里塞大段 payload 文本
- 同时混用多种 supervision 约定而不显式标注 `loss`

如果你有传统对话数据，推荐流程是：

```text
legacy dataset
  -> standalone importer / converter
  -> ACML documents
  -> packaged ACML dataset
  -> train_sft.py
```

## 6. 推荐的数据集目录布局

小型本地实验可以使用：

```text
my-dataset/
  train.jsonl
  valid.jsonl
```

其中每行都是：

```json
{"acml": "<acml version=\"0\">...</acml>"}
```

如果你更偏向“一个样本一个文件”的作者工作流，也可以先这样维护：

```text
my-samples/
  0001.acml
  0002.acml
  0003.acml
```

然后再用一个小脚本把它们打包成：

- `train.jsonl`
- 或 `datasets.Dataset`

也就是说：

- authoring 适合“一个样本一个 `.acml`”
- training packaging 更适合“一个 `acml` 列数据集”

## 7. 打包前检查清单

在把样本喂给训练入口前，建议至少检查：

- 每条 record 都有 `acml` 列
- 每条 `acml` 都能被 parser 正确解析
- 每条训练样本至少有一个 `loss="true"`
- 没有把大段原文错误塞进 attribute
- 角色只使用当前项目支持的集合
- 样本没有明显重复或空 document

## 8. 与 preview / smoke scripts 的关系

当前仓库里最直接的入口是：

- `bash scripts/preview_acml_tiny.sh`
- `bash scripts/train_acml_tiny_smoke.sh`
- `ACML_DATASET_PATH=/path/to/train.acml bash scripts/train_acml_dataset.sh`

这三条脚本分别对应：

- 预览单条最小 ACML 样本
- 用单条最小 ACML 样本跑训练 smoke
- 对真实 ACML 数据集做正式训练

## 9. 独立工具

当前仓库已经提供两个独立工具：

### 9.1 `src/pack_acml_dataset.py`

用途：

- 把“一样本一文件”的 `.acml` authoring 工作区打包成训练可消费的数据集
- 输出 `jsonl`
- 或输出 `datasets` 保存目录

示例：

```bash
python src/pack_acml_dataset.py my-samples/ \
  --output_path train.jsonl
```

```bash
python src/pack_acml_dataset.py my-samples/ \
  --output_path packed_dataset \
  --output_format dataset
```

可选能力：

- `--include_source_path`
- `--allow_unsupervised`

默认会先做基础校验；如果发现坏样本，会拒绝写出结果。

### 9.2 `src/validate_acml_dataset.py`

用途：

- 对 `.acml` 文件、`json/jsonl`、本地 `datasets` 目录或 Hub dataset 逐条做检查
- 检查 `acml` 列、parse、kind、supervision
- 输出摘要统计与前若干条错误

示例：

```bash
python src/validate_acml_dataset.py \
  --dataset_path train.jsonl
```

```bash
python src/validate_acml_dataset.py \
  --dataset_path packed_dataset \
  --dataset_split train
```

可选能力：

- `--allow_unsupervised`
- `--allowed_kinds belief observation me`
- `--limit`

这两个工具都应作为独立辅助工具存在，而不是重新把传统数据兼容逻辑塞回训练主链。
