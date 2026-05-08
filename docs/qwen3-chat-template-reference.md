# Qwen3 对话模板与特殊 Token 完全参考

> 基于本地模型文件 (`/mnt/fast/LLM/Qwen3-1.7B-Base` 和 `/mnt/fast/LLM/Qwen3-1.7B`)、
> Qwen3 技术报告 (arXiv 2505.09388)、sglang/vLLM 推理框架源码、
> Qwen3 官方文档及社区分析综合整理。

---

## 一、架构速览

| 项目 | Qwen3-1.7B-Base | Qwen3-1.7B (Instruct) |
|---|---|---|
| 训练阶段 | 预训练（36T tokens，三阶段） | 预训练 + 四阶段后训练 |
| 参数量 | 1.7B (非嵌入 1.4B) | 1.7B (非嵌入 1.4B) |
| 层数 / 注意力头 | 28 / Q16 KV8 (GQA) | 28 / Q16 KV8 (GQA) |
| 上下文长度 | 32,768 | 40,960 |
| 词表大小 | 151,936 | 151,936 |
| `tokenizer_class` | `Qwen2Tokenizer` | `Qwen2Tokenizer` |
| `eos_token` | `<|endoftext|>` (151643) | `<|im_end|>` (151645) |
| `pad_token` | `<|endoftext|>` (151643) | `<|endoftext|>` (151643) |
| 默认采样 | greedy (`do_sample=false`) | 随机采样 (T=0.6, top_p=0.95, top_k=20) |
| 推理引擎推理 parser | 不需要 | SGLang: `--reasoning-parser qwen3`; vLLM: `--reasoning-parser deepseek_r1` |

---

## 二、基础对话格式：ChatML

Qwen 全系列使用 **ChatML** 格式作为对话骨架，与 OpenAI 的 `im_start`/`im_end` 协议同源：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请介绍一下自己。<|im_end|>
<|im_start|>assistant
你好！我是通义千问...<|im_end|>
```

核心规则：
- **每条消息**以 `<|im_start|>role\n` 开头，以 `<|im_end|>\n` 结尾
- 支持的 role：`system`、`user`、`assistant`、`tool`
- Qwen3 **不再有默认 system prompt**（Qwen2.5 有 "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."）

---

## 三、完整特殊 Token 列表

### 3.1 特殊 Token（`special: true`，tokenizer 硬编码为单 token）

| Token ID | 字符串 | 类别 | 说明 |
|---|---|---|---|
| **151643** | `<|endoftext|>` | 文档分隔 / BOS / EOS | Base 模型用作 eos_token；Instruct 模型仅作 pad_token |
| **151644** | `<|im_start|>` | 对话结构 | 标记一轮对话的开始 |
| **151645** | `<|im_end|>` | 对话结构 | 标记一轮对话的结束；Instruct 模型用作 eos_token |
| 151646 | `<|object_ref_start|>` | 视觉定位 | 目标引用开始 |
| 151647 | `<|object_ref_end|>` | 视觉定位 | 目标引用结束 |
| 151648 | `<|box_start|>` | 视觉定位 | 边界框开始 |
| 151649 | `<|box_end|>` | 视觉定位 | 边界框结束 |
| 151650 | `<|quad_start|>` | 视觉定位 | 四边形开始 |
| 151651 | `<|quad_end|>` | 视觉定位 | 四边形结束 |
| 151652 | `<|vision_start|>` | 多模态 | 视觉输入开始 |
| 151653 | `<|vision_end|>` | 多模态 | 视觉输入结束 |
| 151654 | `<|vision_pad|>` | 多模态 | 视觉填充 |
| 151655 | `<|image_pad|>` | 多模态 | 图像填充 |
| 151656 | `<|video_pad|>` | 多模态 | 视频填充 |

### 3.2 非特殊 Token（`special: false`，可能被分词器拆成多 token）

这些 token 在词表中注册但 `special: false`，意味着分词器不会强制将其映射为单一 token——如果子词拆分能覆盖，就会拆开。

| Token ID | 字符串 | 类别 | 说明 |
|---|---|---|---|
| **151657** | `<tool_call>` | 工具调用 | 函数调用开始标记 |
| **151658** | `</tool_call>` | 工具调用 | 函数调用结束标记 |
| **151659** | `<|fim_prefix|>` | 代码填充 (FIM) | 前缀标记 |
| **151660** | `<|fim_middle|>` | 代码填充 (FIM) | 中间标记 |
| **151661** | `<|fim_suffix|>` | 代码填充 (FIM) | 后缀标记 |
| **151662** | `<|fim_pad|>` | 代码填充 (FIM) | FIM 填充 |
| 151663 | `<|repo_name|>` | 代码仓库 | 仓库名标记 |
| 151664 | `<|file_sep|>` | 代码仓库 | 文件分隔标记 |
| **151665** | `<tool_response>` | 工具调用 | 工具返回开始标记 |
| **151666** | `</tool_response>` | 工具调用 | 工具返回结束标记 |
| **151667** | `<think>` | 推理/思考 | 思考块开始（Qwen3 新增） |
| **151668** | `</think>` | 推理/思考 | 思考块结束（Qwen3 新增） |

> **注意 `<|...|>` vs `<...>` 的区别**（来自 HuggingFace 社区讨论）：
> - `<|im_start|>`、`<|im_end|>` 等使用**尖括号+竖线** (`|`) 包装的是**真正的特殊 token**，tokenizer 将其映射为单一 ID，用作 EOS/BOS 等结构性控制。
> - `<think>`、`</think>`、`<tool_call>` 等使用**XML 风格**尖括号的是**语义占位符**，tokenizer 按普通文本分词（可能拆成多 token），其含义由模型在训练中学习获得，无预设特殊语义。
> - 这种设计混用并非 bug——`<|...|>` 用于底层协议控制，`<...>` 用于高层语义标记。这是 Qwen3 的设计选择。

### 3.3 `additional_special_tokens` 列表（Base 和 Instruct 相同）

```json
[
  "<|im_start|>", "<|im_end|>",
  "<|object_ref_start|>", "<|object_ref_end|>",
  "<|box_start|>", "<|box_end|>",
  "<|quad_start|>", "<|quad_end|>",
  "<|vision_start|>", "<|vision_end|>",
  "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"
]
```

---

## 四、对话模板 (Chat Template) 深度解析

### 4.1 Base 模型 → 无 chat_template？

`Qwen3-1.7B-Base/tokenizer_config.json` **包含完整的 `chat_template`**（Jinja2），与 Instruct 版本几乎完全相同。这是因为：
- Qwen3 预训练数据中**已混入 ChatML 格式的合成指令数据**（技术报告 §3.1）
- tokenizer 从设计之初就内置了 ChatML 控制 token
- Base 模型**可以用** ChatML 格式进行推理，且已有潜在指令跟随能力

### 4.2 Base vs Instruct chat_template 的微小差异

| 差异点 | Base | Instruct |
|---|---|---|
| `content is string` 类型检查 | ❌ 无 | ✅ 有（防止 `content` 为非字符串时模板报错） |
| `enable_thinking=false` 时 prefill | 相同逻辑 | 相同逻辑 |
| 滚动窗口推理保留 | 相同 | 相同 |

实质上两个模板**功能等价**，Instruct 版本仅多了防御性类型检查。

### 4.3 模板渲染输出示例

#### 普通对话（enable_thinking=true，默认）

输入 `messages`：
```json
[
  {"role": "system", "content": "你是一个简洁的助手。"},
  {"role": "user", "content": "解释 SFT"}
]
```

渲染输出（`add_generation_prompt=true`）：
```
<|im_start|>system
你是一个简洁的助手。<|im_end|>
<|im_start|>user
解释 SFT<|im_end|>
<|im_start|>assistant
```

模型在 `<|im_start|>assistant\n` 之后开始自回归生成。由于 `enable_thinking=true`（默认），模型**可以自主决定**是否先输出 `<think>...</think>` 推理块，再输出最终回答。

#### 禁用思考（enable_thinking=false）

渲染输出（`add_generation_prompt=true, enable_thinking=false`）：
```
<|im_start|>system
你是一个简洁的助手。<|im_end|>
<|im_start|>user
解释 SFT<|im_end|>
<|im_start|>assistant
<think>

</think>

```

关键机制：模板在 `assistant` 角色标记后 **prefill 了空 `<think>\n\n</think>\n\n`**，这告诉模型"思考阶段已经结束（空），直接输出回答"。模型看到这个 prefill 后，会跳过推理直接生成最终内容。

> **这是 Qwen3 最巧妙的设计**：不是通过控制 token 或改变模型行为来禁用思考，而是通过**文本预填充**让模型"看到思考已完成"。

### 4.4 软开关：`/think` 和 `/no_think`

当 `enable_thinking=true`（默认），用户可以在消息中动态控制：

```
用户: 请解这道微积分题 /think
助手: <think>设 u = x²，则 du = 2x dx...</think>答案是 ...

用户: 1+1 等于几？ /no_think
助手: 等于 2。
```

规则：
- `/think` 和 `/no_think` 可放在 user 消息或 system 消息中
- 多轮对话中以**最近一条**指令为准
- 这些标记在 chat_template 层面不做特殊解析——模型在训练中学会了遵守它们

### 4.5 滚动窗口推理保留机制

Qwen3 的 chat_template 包含一个精巧的上下文管理逻辑：

```
遍历消息列表（倒序）→ 找到最后一个非 tool_response 的 user 消息的索引
→ 该索引之后的所有 assistant 回复 → 保留完整 <think> 块
→ 该索引之前的 assistant 回复 → 剥离 <think> 块，只保留最终内容
```

效果：
- ✅ 多步工具调用中保留当前任务的推理链
- ✅ 早轮推理被压缩，节省上下文
- ✅ 防止"陈旧的推理"污染新任务

### 4.6 工具调用格式

Qwen3 使用 **Hermes 风格**工具调用模板：

```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "get_weather", "description": "获取天气", "parameters": {...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
北京今天天气怎么样？<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "北京"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 22, "condition": "晴"}
</tool_response><|im_end|>
<|im_start|>assistant
北京今天晴天，气温 22°C。<|im_end|>
```

- 工具返回以 `role: "tool"` 格式传入，模板会自动转换为 `<|im_start|>user\n<tool_response>...</tool_response><|im_end|>`
- SGLang 部署时使用 `--tool-call-parser qwen`（或 `qwen3_coder` 用于 Qwen3-Coder 系列）

---

## 五、推理框架集成

### 5.1 SGLang

```bash
# 启动服务
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen

# 请求时禁用思考
curl ... -d '{
  "model": "Qwen/Qwen3-1.7B",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

- `--reasoning-parser qwen3`：解析 `<think>...</think>` 块，分离到 `reasoning_content` 字段
- `--reasoning-parser qwen3-thinking`：用于 Qwen3-Thinking-2507 等纯推理模型
- SGLang 的 `template_detection.py` 通过检测模板中的 `enable_thinking` toggle 参数且 `default_enabled=true` 来识别 Qwen3 系列

### 5.2 vLLM

```bash
vllm serve Qwen/Qwen3-1.7B \
  --enable-reasoning \
  --reasoning-parser deepseek_r1
```

- vLLM 使用 `deepseek_r1` 解析器处理 Qwen3 的 `<think>` 块（因为格式相同）
- 同样支持通过 `chat_template_kwargs` 传递 `enable_thinking`

---

## 六、Base vs Instruct 关键差异总结

| 维度 | Qwen3-1.7B-Base | Qwen3-1.7B (Instruct) |
|---|---|---|
| **预训练** | ✅ 36T tokens，三阶段 | ✅ 同 Base |
| **Long-CoT 冷启动 SFT** | ❌ | ✅ stage 1 |
| **推理 RL (GRPO)** | ❌ | ✅ stage 2 |
| **思维模式融合 SFT** | ❌ | ✅ stage 3 |
| **通用 RL 对齐** | ❌ | ✅ stage 4 |
| **`/think` `/no_think` 切换** | ❌ 不会响应 | ✅ 支持软切换 |
| **`enable_thinking` 机制** | ⚠️ chat_template 中有，但模型未训练 | ✅ 完整支持 |
| **ChatML 格式理解** | ✅ 预训练中已学会 | ✅ 熟练 |
| **基本问答能力** | ✅ 给 ChatML 格式即可 | ✅ 熟练 |
| **安全对齐 / 拒答** | ❌ 未做 | ✅ RL stage 4 |
| **数学推理链** | ⚠️ 有基础能力（预训练 Stage 2 强化） | ✅ RL 强化后更强 |
| **默认系统提示** | 无（与 Instruct 一致） | 无 |
| **eos_token** | `<|endoftext|>` (151643) | `<|im_end|>` (151645) |

### 关键启示

1. **Base 不是传统意义的"纯续写"模型**：预训练数据中已包含 ChatML 格式合成数据，模型具备潜在对话能力
2. **Base 的 chat_template 存在但模型未做对应训练**：`enable_thinking=false` 的 prefill 机制在 Base 上可能不稳定
3. **做对话模板改造实验时**：Base 模型适合研究"模板格式对模型行为的影响"，Instruct 模型适合研究"思考/非思考模式切换"

---

## 七、在 study-sft 项目中的建议

### 7.1 你当前的 ChatML 格式

项目 `src/study_sft/formats.py` 使用的格式：
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

这与 Qwen3 官方的 ChatML 格式**完全兼容**。你的 5-step LoRA 实验之所以能得到合理回答，正是因为：
- 预训练中的 ChatML 数据已教会模型这种格式
- 3 条 alpaca 样本的 LoRA 做了极轻量的"格式确认"

### 7.2 模板改造建议

如果要进行对话模板改造实验，可以尝试：

| 实验方向 | 具体做法 |
|---|---|
| **Late System** | 将 system 消息移到 user 之后（你项目已有 `late_system` 模式） |
| **无 system 提示** | 去掉 system 消息，直接 user/assistant 对话 |
| **多轮对话** | 使用 `messages` 格式传入完整历史 |
| **Thinking 注入** | 在 assistant 回复前 prefill `<think>\n\n</think>\n\n` |
| **工具调用格式** | 引入 `<tool_call>` / `<tool_response>` 结构 |
| **自定义 role** | 测试模型对非标准 role 的泛化（如你的 `belief` / `observation`） |

### 7.3 eos_token 陷阱

- Base 模型的 `eos_token_id=151643` (`<|endoftext|>`)
- Instruct 模型的 `eos_token_id=151645` (`<|im_end|>`)
- 你的 LoRA adapter 保存的 tokenizer_config 中 `eos_token="<\|endoftext\|>"`（与 Base 一致）
- **做 SFT 时**：如果训练数据以 `<|im_end|>` 结尾，建议将 eos_token 设为 `<|im_end|>`，否则模型可能不会主动停止生成

---

## 八、参考资料

- Qwen3 技术报告: https://arxiv.org/abs/2505.09388
- Qwen3 官方博客: https://qwenlm.github.io/blog/qwen3/
- Qwen3 GitHub: https://github.com/QwenLM/Qwen3
- Qwen3 官方文档: https://qwen.readthedocs.io/en/latest/
- SGLang Qwen3 推理文档: `--reasoning-parser qwen3`
- HuggingFace Blog「The 4 Things Qwen-3's Chat Template Teaches Us」: https://huggingface.co/blog/cfahlgren1/4-things-qwen3-chat-template
- ModelScope 模型页: https://modelscope.cn/models/Qwen/Qwen3-1.7B-Base
