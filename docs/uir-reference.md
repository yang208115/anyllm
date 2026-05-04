# UIR 模型参考

本文档是 AnyLLM Universal Intermediate Representation (UIR) 所有数据模型的完整字段参考。

所有模型均基于 Pydantic v2，位于 `anyllm.schema` 包中。

## UniversalRequest

**模块**：`anyllm.schema.request`

统一请求体，是网关和转换器的唯一内部流通格式。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `version` | `str` | `"uai.v1"` | UIR 版本号，用于兼容性检测 |
| `model` | `ModelRef` | **必填** | 目标模型引用 |
| `messages` | `List[Message]` | **必填** | 对话历史，按时间顺序排列 |
| `instructions` | `List[ContentBlock]` | `[]` | System Prompt，抽离至顶层 |
| `tools` | `List[ToolDef]` | `[]` | 可调用的工具列表 |
| `tool_choice` | `ToolChoice \| None` | `None` | 工具选择策略 |
| `response_format` | `ResponseFormat \| None` | `None` | 结构化输出格式 |
| `generation` | `GenerationConfig` | `GenerationConfig()` | 生成参数 |
| `stream` | `bool` | `False` | 是否启用流式响应 |
| `state` | `ConversationState` | `ConversationState()` | 有状态会话信息（L2） |
| `metadata` | `Dict[str, Any]` | `{}` | 用户自定义元数据（不传给 provider） |
| `vendor` | `Dict[str, Any]` | `{}` | 厂商特定字段透传区 |

### instructions 字段

System Prompt 被抽离至顶层，而非作为第一条 `system` 角色消息。这是因为不同 provider 对 system prompt 的位置有不同约定：

| Provider | 映射方式 |
|----------|---------|
| OpenAI Chat | 转为首条 `{"role": "system", "content": str}` |
| OpenAI Responses | 转为 `instructions` 字段 |
| Anthropic | 转为顶层 `system` 字段 |
| Gemini | 转为 `systemInstruction.parts` |
| Bedrock | 转为顶层 `system[{text: ...}]` |

### vendor 字段

按 provider 名称分组的透传区，适配器在 `uir_to_request` 时将对应部分合并到输出：

```python
request = UniversalRequest(
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    messages=[...],
    vendor={
        "anthropic": {
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        },
        "openai": {
            "store": True,
        },
    },
)
```

---

## ModelRef

**模块**：`anyllm.schema.request`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | **必填** | 模型名称，如 `"gpt-4o"`、`"claude-sonnet-4-5"` |
| `provider` | `ProviderName` | `"unknown"` | 来源 provider，用于路由 |
| `raw` | `Dict[str, Any]` | `{}` | provider 原始模型元数据 |

`ProviderName` 可选值：`"openai"` | `"anthropic"` | `"google"` | `"bedrock"` | `"ollama"` | `"cloudflare"` | `"unknown"`

---

## GenerationConfig

**模块**：`anyllm.schema.request`

| 字段 | 类型 | 默认值 | 说明 | Provider 支持 |
|------|------|--------|------|--------------|
| `temperature` | `float \| None` | `None` | 采样温度 | 全部 |
| `top_p` | `float \| None` | `None` | Nucleus sampling | 全部 |
| `top_k` | `int \| None` | `None` | Top-K sampling | Anthropic / Gemini / Bedrock |
| `max_output_tokens` | `int \| None` | `None` | 最大输出 tokens | 全部（Anthropic 必填） |
| `stop` | `List[str] \| None` | `None` | 停止序列 | 全部 |
| `seed` | `int \| None` | `None` | 随机种子 | OpenAI / Gemini |
| `presence_penalty` | `float \| None` | `None` | 存在惩罚 | OpenAI |
| `frequency_penalty` | `float \| None` | `None` | 频率惩罚 | OpenAI |
| `raw` | `Dict[str, Any]` | `{}` | 额外参数透传区 | — |

---

## ConversationState

**模块**：`anyllm.schema.request`

有状态会话信息，L2 能力，仅 OpenAI Responses/Assistants API 支持。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `conversation_id` | `str \| None` | `None` | Responses API 会话 ID |
| `thread_id` | `str \| None` | `None` | Assistants API Thread ID |
| `run_id` | `str \| None` | `None` | Assistants API Run ID |
| `previous_response_id` | `str \| None` | `None` | Responses API 上一轮响应 ID |
| `provider_state` | `Dict[str, Any]` | `{}` | provider 特定状态数据 |

---

## Message

**模块**：`anyllm.schema.message`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `role` | `Role` | **必填** | 消息角色 |
| `content` | `List[ContentBlock]` | `[]` | 多模态内容块列表 |
| `id` | `str \| None` | `None` | 消息 ID |
| `name` | `str \| None` | `None` | 发言者名称 |
| `tool_calls` | `List[ToolCall]` | `[]` | 工具调用列表（顶层表达） |
| `tool_results` | `List[ToolResult]` | `[]` | 工具结果列表（顶层表达） |
| `provider` | `Dict[str, Any]` | `{}` | 厂商原始数据透传区 |

`Role` 可选值：`"system"` | `"developer"` | `"user"` | `"assistant"` | `"tool"`

### tool_calls 与 ToolCallBlock 的关系

不同 provider 表达工具调用的位置不同：

| 表达方式 | 适用 provider | 说明 |
|----------|--------------|------|
| `Message.tool_calls[]` | OpenAI Chat / Ollama | 工具调用在消息顶层 |
| `Message.content[ToolCallBlock]` | Anthropic / Gemini / Bedrock | 工具调用嵌入 content |

UIR 同时支持两种表达，适配器在转换时互相派生。如果同一个 ToolCall 同时出现在 `tool_calls` 和 `content` 中，适配器会去重。

### 快捷构造方法

| 方法 | 作用 |
|------|------|
| `Message.user_text(text)` | 纯文本 user 消息 |
| `Message.assistant_text(text)` | 纯文本 assistant 消息 |
| `Message.tool_result(call_id, text, is_error=False)` | 工具结果消息 |

---

## ContentBlock 体系

**模块**：`anyllm.schema.content`

`ContentBlock` 是 9 种内容块的联合类型，通过 Pydantic Discriminated Union 按 `type` 字段自动路由反序列化。

### TextBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["text"]` | `"text"` | — |
| `text` | `str` | **必填** | 文本内容 |
| `annotations` | `List[Dict]` | `[]` | OpenAI Responses API annotations |

### ImageBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["image"]` | `"image"` | — |
| `source` | `MediaSource` | **必填** | 图片来源 |
| `detail` | `"low" \| "high" \| "auto" \| None` | `None` | 分辨率控制（OpenAI Vision 专用） |

### AudioBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["audio"]` | `"audio"` | — |
| `source` | `MediaSource` | **必填** | 音频来源 |
| `format` | `str \| None` | `None` | 音频编码格式（mp3/wav/opus） |

### FileBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["file"]` | `"file"` | — |
| `source` | `MediaSource` | **必填** | 文件来源 |
| `mime_type` | `str \| None` | `None` | MIME 类型 |
| `filename` | `str \| None` | `None` | 文件名 |

### ThinkingBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["thinking"]` | `"thinking"` | — |
| `text` | `str \| None` | `None` | 明文推理内容 |
| `encrypted` | `str \| None` | `None` | Anthropic 加密形式 |
| `signature` | `str \| None` | `None` | Anthropic 签名 |

### RefusalBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["refusal"]` | `"refusal"` | — |
| `text` | `str` | **必填** | 拒绝原因 |

### ToolCallBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["tool_call"]` | `"tool_call"` | — |
| `call` | `ToolCall` | **必填** | 工具调用详情 |

### ToolResultBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["tool_result"]` | `"tool_result"` | — |
| `result` | `ToolResult` | **必填** | 工具结果详情 |

### ProviderBlock

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `Literal["provider_block"]` | `"provider_block"` | — |
| `provider` | `str` | **必填** | 来源 provider 标识符 |
| `value` | `Any` | **必填** | 原始数据 |

---

## MediaSource

**模块**：`anyllm.schema.content`

统一媒体来源描述符，支持 4 种 kind。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kind` | `"url" \| "base64" \| "file_id" \| "bytes"` | **必填** | 来源类型 |
| `value` | `Any` | **必填** | 实际数据 |
| `mime_type` | `str \| None` | `None` | MIME 类型 |
| `filename` | `str \| None` | `None` | 文件名 |

不同 kind 对应的 value 类型：

| kind | value 类型 | 示例 |
|------|-----------|------|
| `url` | `str` | `"https://example.com/image.png"` |
| `base64` | `str` | Base64 编码字符串（不含 `data:` 前缀） |
| `file_id` | `str` | `"file-abc123"` |
| `bytes` | `bytes` | 原始字节对象 |

---

## ToolCall

**模块**：`anyllm.schema.content`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | `str` | **必填** | 调用唯一 ID |
| `name` | `str` | **必填** | 工具名称 |
| `arguments` | `Any` | **必填** | 已解析的参数（dict） |
| `raw_arguments` | `str \| None` | `None` | 原始 JSON 字符串（调试用） |
| `provider` | `Dict[str, Any]` | `{}` | 厂商数据透传区 |

### parse_tool_arguments

工具函数，规范化 tool_call arguments：

```python
from anyllm import parse_tool_arguments

# 字符串 JSON → dict
args, raw, warning = parse_tool_arguments('{"city": "Tokyo"}')
# args = {"city": "Tokyo"}, raw = '{"city": "Tokyo"}', warning = None

# 已经是 dict → 直接返回
args, raw, warning = parse_tool_arguments({"city": "Tokyo"})
# args = {"city": "Tokyo"}, raw = None, warning = None

# 非法 JSON → 返回空 dict + warning code
args, raw, warning = parse_tool_arguments("not json")
# args = {}, raw = "not json", warning = "INVALID_TOOL_ARGUMENTS_JSON"
```

---

## ToolResult

**模块**：`anyllm.schema.content`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `call_id` | `str` | **必填** | 对应 ToolCall.id |
| `content` | `List[ContentBlock]` | **必填** | 结果内容（支持富文本） |
| `name` | `str \| None` | `None` | 工具名称（Gemini 需要） |
| `is_error` | `bool` | `False` | 是否为错误结果 |
| `provider` | `Dict[str, Any]` | `{}` | 厂商数据透传区 |

---

## ToolDef

**模块**：`anyllm.schema.tools`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `"function" \| "builtin" \| "mcp" \| "provider"` | `"function"` | 工具类型 |
| `name` | `str` | **必填** | 工具名称（唯一，建议 snake_case） |
| `description` | `str \| None` | `None` | 功能描述 |
| `input_schema` | `Dict \| None` | `None` | 输入参数 JSON Schema |
| `output_schema` | `Dict \| None` | `None` | 输出结果 JSON Schema |
| `provider` | `Dict[str, Any]` | `{}` | 厂商特定参数 |

---

## ToolChoice

**模块**：`anyllm.schema.tools`

Discriminated Union，支持 4 种策略：

| 类型 | `type` 值 | 说明 |
|------|-----------|------|
| `AutoToolChoice` | `"auto"` | 模型自动决定 |
| `NoneToolChoice` | `"none"` | 禁止调用工具 |
| `RequiredToolChoice` | `"required"` | 必须调用至少一个 |
| `SpecificToolChoice` | `"tool"` | 必须调用指定工具（需提供 `name`） |

---

## ResponseFormat

**模块**：`anyllm.schema.tools`

Discriminated Union，支持 3 种格式：

| 类型 | `type` 值 | 说明 |
|------|-----------|------|
| `TextResponseFormat` | `"text"` | 纯文本（默认） |
| `JsonObjectResponseFormat` | `"json_object"` | JSON 对象（无 schema 约束） |
| `JsonSchemaResponseFormat` | `"json_schema"` | 带 schema 约束的 JSON（Structured Outputs） |

`JsonSchemaResponseFormat` 额外字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | schema 名称 |
| `json_schema` | `Dict` | JSON Schema（API 中序列化为 `"schema"` 键名） |
| `strict` | `bool` | 是否启用 strict mode（OpenAI 专用） |

---

## UniversalResponse

**模块**：`anyllm.schema.response`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | `str \| None` | `None` | 响应 ID |
| `model` | `str \| None` | `None` | 实际使用的模型名称 |
| `output` | `List[Message]` | `[]` | 输出消息列表 |
| `stop_reason` | `StopReason` | `"unknown"` | 停止原因 |
| `usage` | `Usage \| None` | `None` | Token 用量 |
| `state` | `ConversationState` | `ConversationState()` | 会话状态 |
| `raw` | `Any` | `None` | 原始响应数据 |
| `vendor` | `Dict[str, Any]` | `{}` | 厂商特定响应字段 |

`StopReason` 可选值：`"end_turn"` | `"max_tokens"` | `"stop_sequence"` | `"tool_calls"` | `"content_filter"` | `"error"` | `"unknown"`

### normalize_stop_reason

将 provider 原始 stop reason 映射为统一值：

| Provider | 原始值 | UIR 映射 |
|----------|--------|---------|
| OpenAI | `"stop"` | `"end_turn"` |
| OpenAI | `"length"` | `"max_tokens"` |
| OpenAI | `"tool_calls"` | `"tool_calls"` |
| OpenAI | `"content_filter"` | `"content_filter"` |
| Anthropic | `"end_turn"` | `"end_turn"` |
| Anthropic | `"max_tokens"` | `"max_tokens"` |
| Anthropic | `"tool_use"` | `"tool_calls"` |
| Gemini | `"STOP"` | `"end_turn"` |
| Gemini | `"MAX_TOKENS"` | `"max_tokens"` |
| Gemini | `"SAFETY"` | `"content_filter"` |

---

## Usage

**模块**：`anyllm.schema.usage`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_tokens` | `int \| None` | `None` | 输入 token 数 |
| `output_tokens` | `int \| None` | `None` | 输出 token 数 |
| `total_tokens` | `int \| None` | `None` | 总 token 数 |
| `reasoning_tokens` | `int \| None` | `None` | 推理 token 数（含在 output 中） |
| `cached_input_tokens` | `int \| None` | `None` | 缓存命中 token 数（含在 input 中） |
| `provider` | `Dict[str, Any]` | `{}` | 厂商原始用量数据 |

---

## UniversalStreamEvent

**模块**：`anyllm.schema.stream`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `StreamEventType` | **必填** | 事件类型 |
| `response_id` | `str \| None` | `None` | 响应 ID |
| `message_id` | `str \| None` | `None` | 消息 ID |
| `index` | `int \| None` | `None` | 内容块索引 |
| `delta` | `ContentBlock \| None` | `None` | 内容增量 |
| `tool_call` | `ToolCall \| None` | `None` | 工具调用信息 |
| `usage` | `Usage \| None` | `None` | Token 用量 |
| `raw` | `Any` | `None` | 原始 event 数据 |

`StreamEventType` 可选值：

| 值 | 说明 |
|----|------|
| `"response_started"` | 响应开始 |
| `"message_started"` | 新消息块开始 |
| `"content_delta"` | 文本/图片增量 |
| `"tool_call_started"` | 工具调用开始 |
| `"tool_call_delta"` | 工具调用参数增量 |
| `"tool_call_completed"` | 工具调用完整 |
| `"message_completed"` | 消息块结束 |
| `"response_completed"` | 响应结束 |
| `"error"` | 错误 |

---

## ConversionWarning

**模块**：`anyllm.schema.warnings`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `code` | `str` | **必填** | 机器可读的错误码 |
| `path` | `str` | **必填** | 触发警告的字段路径 |
| `message` | `str` | **必填** | 人类可读的描述 |
| `severity` | `"info" \| "warning" \| "error"` | `"warning"` | 严重程度 |

详细的 warning code 列表见 [转换警告参考](warnings.md)。

---

## ConversionResult\[T\]

**模块**：`anyllm.schema.warnings`

所有转换方法的统一返回类型。

| 字段/方法 | 类型 | 说明 |
|-----------|------|------|
| `value` | `T` | 转换结果 |
| `warnings` | `List[ConversionWarning]` | 警告列表 |
| `has_errors` | `bool`（property） | 是否含 error 级别警告 |
| `add_warning(...)` | method | 追加一条警告 |
| `merge_warnings(other)` | method | 合并另一个 Result 的警告 |
