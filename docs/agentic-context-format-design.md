# Agentic Context Format Design

> 状态：当前 `v0` 方向的设计说明。本文档负责说明层次、边界与当前收敛结论；不再兼做感性手稿或 worked example。

## 文档分工

- [README.md](../README.md)：仓库入口、快速开始、相关文档链接。
- [examples/agentic-ml/01-ask-and-answer.txt](../examples/agentic-ml/01-ask-and-answer.txt)：最早的概念手稿，保留为历史/附录，不作为当前规范。
- [examples/xml-authoring/01-code-review-and-patch.xml](../examples/xml-authoring/01-code-review-and-patch.xml)：`XML authoring` 的 worked example，用来帮助人和 LLM Agent 直观看格式，不承担规范定义。
- 本文：说明三层模型、`v0` 约束、实现边界与待决问题。

若不同文件之间出现表述冲突，以本文为准。

## 1. 背景

这个仓库当前的主线，不再是并行维护多套 prompt 文本模板，而是把训练、预览、推理统一到一个共享的 agentic context 协议上。

相较于传统 `system / user / assistant` 聊天抽象，这里更关心 Agent 运行时的三类语义：

- `observation`：来自用户、终端、网页、文件、工具结果等外部可感知输入。
- `belief`：模型当前应当参考的内部状态、关系、约束、工具表或运行模式。
- `me`：模型当前这一步需要产出的主体输出。

目标不是换一套提示词外观，而是建立一个更适合作为训练与推理中间表示的上下文协议。

## 2. 三层模型

当前设计明确分成三层，每层承担不同职责。

### 2.1 Authoring Syntax

这是给人和 LLM Agent 编写样本时使用的文本形式。

要求：

- 可读、可编辑。
- 能自然表达结构化内容和 `opaque_payload` 叶子。
- 不直接暴露最终 wire format 的 special token。

当前探索方向是 XML authoring，因为 mixed content 和嵌套参数在 XML 中通常比 JSON 更自然。本文不冻结 XML 语法细节，只要求它能无损映射到 canonical typed IR，而不直接定义训练时看到的 wire format。

### 2.2 Canonical Typed IR

这是仓库内部的规范真相源。

当前 `v0` 的核心形状可以概括为：

- `context.messages: [message, ...]`
- `message = { role, loss?, content: [opaque_payload, ...] }`
- `opaque_payload = { text }`

这里的关键约束是：

- `content` 仍然是序列，所以单条 `message` 可以承载多个 payload。
- `opaque_payload` 是叶子节点，不在其内部继续承载协议结构。
- 当前 `v0` 不再把匿名通用嵌套 AST 当成核心能力。

### 2.3 Encoded Protocol

这是 encoder 最终产出的 token id 序列，也是训练与推理直接消费的形式。

顶层仍复用 ChatML 风格消息边界：

```text
<|im_start|>role
...message content...
<|im_end|>
```

当前 `v0` 只在消息内部使用 `opaque_payload` framing：

```text
<|box_start|>...escaped payload text...<|box_end|>
```

这里的 special token 只应该由可信 serializer 插入，不应该出现在 authoring 层。

## 3. 当前 v0 收敛结论

本仓库当前代码与文档应围绕以下结论保持一致。

### 3.1 角色语义

| role | 当前语义 | 近似类比 |
|---|---|---|
| `observation` | 外部观察、材料、用户输入、工具返回 | `user` |
| `belief` | 内部状态、规则、关系、工具表、运行模式 | `system` |
| `me` | 当前步骤的模型输出目标 | `assistant` |

### 3.2 `message` 内容形状

`v0` 允许一个 `message` 内有多个 `opaque_payload`，这对多文件、多文档、多段规则都很重要。

但 `v0` 也刻意收窄了边界：

- 不把“匿名递归结构节点”作为当前 canonical schema 的一部分。
- 不让 `opaque_payload` 本身再嵌结构。
- 不让作者直接书写 special-token 混合串。

如果后续确实需要 richer tree，方向也应是“显式语义节点”，例如文件、网页、脚本参数等，而不是回到匿名 `structured_region`。

### 3.3 `loss` 语义

训练目标由 message 级 `loss` 显式标记，不按 role 猜测。

这意味着：

- `observation` 和 `belief` 默认不参与监督。
- 历史 `me` 可以只是上下文。
- 当前目标 `me` 是否参与训练，由样本构造层明确写入 `loss: true`。

## 4. 安全与序列化不变量

当前 `v0` 最重要的不变量，不是文本层“看起来像有边界”，而是 token id 层的结构完整性。

### 4.1 Reserved token 只能来自可信 serializer

必须建立以下约束：

```text
trusted serializer 可以插入 reserved structure token id
untrusted payload encoder 不能产生 reserved structure token id
```

因此：

- 作者不应该手写 `<|im_start|>`、`<|im_end|>`、`<|box_start|>`、`<|box_end|>` 来表达结构。
- `opaque_payload` 里的字面 special-token 字符串，只能被当作普通文本内容处理。

### 4.2 `opaque_payload` 是不可信文本入口

`opaque_payload` 用来承载用户原文、工具输出、文件片段、网页内容等不可信文本。

实现层需要保证：

- payload 文本进入安全 encoder。
- 编码结果不包含任何 reserved ids。
- validate/parser 看到的结构边界只来自可信 serializer。

### 4.3 Authoring Syntax 不是 Wire Format

无论作者写的是 XML 还是别的文本形式，authoring 层都只是“好写、好读、好审阅”的表示。

真正参与训练、推理、验证的仍然是：

```text
authoring syntax
  -> typed IR
  -> encoded protocol
```

## 5. XML Authoring 的当前定位

`XML authoring` 目前的职责非常明确：

- 服务样本撰写与讨论。
- 演示如何表达结构节点与 payload 叶子。
- 给未来 parser / serializer 提供直观的 user story。

但它现在还不是冻结的正式规范。

当前更合理的态度是：

- 用 XML 做 worked example 和 authoring exploration。
- 用 typed IR 作为内部真相源。
- 等 XML 上的几个关键约束稳定后，再把它规格化成更严格的 authoring 文档。

### 5.1 当前建议的 authoring 原则

- 结构语义尽量由显式元素名表达，而不是匿名盒子。
- 大块、不可信、可能包含 special-token 字面串的文本，都放进显式 payload 叶子。
- 不让 attribute 承载大段不可信字符串，尤其是脚本片段、文件内容、`old_text` / `new_text` 这类值。
- XML 示例可以为了可读性使用 `CDATA`，但 `CDATA` 不是语义层特性。

### 5.2 Worked Example 的职责边界

[examples/xml-authoring/01-code-review-and-patch.xml](../examples/xml-authoring/01-code-review-and-patch.xml) 只负责展示：

- 角色如何出现。
- 单条 message 内多个 payload 如何出现。
- 结构节点与 payload 叶子如何配合。
- payload 中的字面 special-token 如何保持为普通文本。

它不再额外承担“为什么这样设计”的长篇解释。

## 6. README、手稿与规范的关系

为了避免同一概念在多个地方重复描述，推荐按下面的阅读顺序理解仓库：

1. 先看 [README.md](../README.md)，了解仓库目标、快速开始和关键入口。
2. 再看本文，理解三层模型和当前 `v0` 边界。
3. 需要历史直觉时，再看 [examples/agentic-ml/01-ask-and-answer.txt](../examples/agentic-ml/01-ask-and-answer.txt)。
4. 需要直观样例时，再看 [examples/xml-authoring/01-code-review-and-patch.xml](../examples/xml-authoring/01-code-review-and-patch.xml)。

也就是说：

- README 不再展开讲协议细节。
- 手稿不再承担规范责任；其中保留的 `quad/script` 等早期想法不应被视为当前 `v0` 承诺。
- worked example 不再承担设计论证。
- 本文承担术语、边界与当前收敛方向。

## 7. 当前实现映射

与本文最相关的当前代码层大致如下：

- `src/study_sft/agentic_context_model.py`：canonical typed IR。
- `src/study_sft/agentic_context_schema.py`：外部对象到 typed IR 的适配入口。
- `src/study_sft/agentic_context.py`：typed IR 到 encoded protocol 的 encoder、validation 与相关 rich result。
- `src/study_sft/samples.py`：把 `alpaca/messages/sharegpt` 等格式投影成共享 conversation/context。
- `src/study_sft/training_data.py`：训练特征、label 与截断。

这里真正要稳定的是边界，而不是某个具体的 authoring 文本外观。

## 8. 非目标

当前阶段明确不做这些事：

- 不把 XML 直接当成模型最终看到的 wire format。
- 不要求手稿、设计文档、XML 示例三处同时维护完整规范。
- 不把尚未确定的 richer AST 形状提前承诺为 `v0` 能力。
- 不把安全性建立在“模型会理解文本说明”上，而是建立在 serializer / validator / parser 边界上。

## 9. 待后续规格化的问题

当前仍值得继续打磨，但尚未冻结的点主要有：

- XML authoring 里哪些元素允许 mixed content，哪些必须显式 `<payload>` 包裹。
- XML authoring 的标签集应偏“工具语义”还是偏“更抽象的结构语义”。
- 当后续需要 richer tree 时，typed IR 应新增哪些显式节点，而不是重新引入匿名容器。
- debug/inspect 视图应在多大程度上独立于 core encoder。

这些问题会在后续 authoring 设计和解析器实现时继续收敛；本文先把当前分层和职责边界固定住。
