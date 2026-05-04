# 转换警告参考

本文档列出 AnyLLM 转换过程中可能产生的所有 `ConversionWarning` 错误码，包括触发条件、严重等级和推荐处理策略。

## 概述

AnyLLM 的所有转换方法返回 `ConversionResult[T]`，其中包含一个 `warnings` 列表。每个 warning 包含：

| 字段 | 说明 |
|------|------|
| `code` | 机器可读的错误码（本文档按此分类） |
| `path` | 触发警告的字段路径（如 `"messages[2].content[0]"`） |
| `message` | 人类可读的中文描述 |
| `severity` | 严重等级：`info` / `warning` / `error` |

### 严重等级

| 等级 | 含义 | 影响 |
|------|------|------|
| `info` | 纯透传/提示 | 不影响功能，仅记录 |
| `warning` | 字段降级 | 目标请求可能丢失部分语义 |
| `error` | 无法正确转换 | 目标请求可能不可用 |

### 读取 Warnings

```python
result = converter.convert_request("openai_chat", "anthropic", request)

# 检查是否有错误级别的警告
if result.has_errors:
    print("转换有严重问题！")

# 遍历所有警告
for w in result.warnings:
    print(f"[{w.severity}] {w.code} at {w.path}: {w.message}")

# 按 severity 过滤
errors = [w for w in result.warnings if w.severity == "error"]
warnings = [w for w in result.warnings if w.severity == "warning"]
infos = [w for w in result.warnings if w.severity == "info"]
```

---

## 模态不兼容

### UNSUPPORTED_MODALITY

**等级**：`warning`

**触发条件**：目标 provider 不支持请求中的某种内容模态。

| 场景 | 说明 |
|------|------|
| ImageBlock → 纯文本 provider | 图片降级为 `"[Image omitted]"` |
| AudioBlock → 不支持音频的 provider | 音频降级为 `"[Audio omitted]"` |

**处理策略**：检查目标 provider 的 `ProviderCapabilities`，确保其支持请求中用到的模态。或在发送前移除不支持的内容块。

### IMAGE_SOURCE_DOWNGRADED

**等级**：`warning`

**触发条件**：图片来源格式不被目标 provider 支持。

| 场景 | 说明 |
|------|------|
| URL 图片 → Anthropic（仅支持 base64） | 建议使用 `ImageResolutionInterceptor` 预转换 |
| file_id 图片 → 不支持 file_id 的 provider | 降级为占位文本 |

**处理策略**：注册 `ImageResolutionInterceptor`，它会自动将 URL 图片下载并转为 base64。

### FILE_REFERENCE_NOT_SUPPORTED

**等级**：`warning`

**触发条件**：目标 provider 不支持文件引用（`FileBlock`）。

**处理策略**：文件引用降级为占位文本 `"[File omitted: filename]"`。如果文件内容是必需的，考虑先提取文件内容为文本。

---

## 角色和消息结构

### ROLE_DOWNGRADED

**等级**：`warning`

**触发条件**：消息角色被合并或降级。

| 场景 | 说明 |
|------|------|
| `system` 消息 → Anthropic | Anthropic 不支持 system 角色消息，合并到顶层 system 字段 |
| `developer` 消息 → Anthropic | 同上 |
| `tool` 消息 → Anthropic | 合并到 user 消息的 content block 中 |

**处理策略**：这通常是正常行为，`RoleConsolidationInterceptor` 会自动处理。

### UNSUPPORTED_ROLE

**等级**：`warning`

**触发条件**：目标 provider 完全不支持某个角色。

**处理策略**：检查角色映射逻辑，确保角色转换正确。

---

## 工具调用

### INVALID_TOOL_ARGUMENTS_JSON

**等级**：`warning`

**触发条件**：工具调用的 `arguments` 字段不是有效的 JSON 字符串。

```python
# 触发示例：模型返回了非法 JSON
tool_call = {
    "id": "call_1",
    "function": {
        "name": "get_weather",
        "arguments": "not valid json{",  # 无法解析
    },
}
```

**处理策略**：检查 `ToolCall.raw_arguments` 字段获取原始字符串。arguments 字段会被设为空 dict `{}`。

### TOOL_CALL_BLOCK_DOWNGRADED

**等级**：`warning`

**触发条件**：`ToolCallBlock` 无法以目标 provider 的格式表达，降级为占位文本。

**处理策略**：通常不需要处理，适配器会自动将 `ToolCallBlock` 转换为顶层 `tool_calls`（OpenAI 风格）或 content block（Anthropic 风格）。

### TOOL_RESULT_BLOCK_DOWNGRADED

**等级**：`warning`

**触发条件**：`ToolResultBlock` 无法以目标 provider 的格式表达，降级为占位文本。

**处理策略**：通常不需要处理，适配器会自动处理格式转换。

### TOOL_CHOICE_DOWNGRADED

**等级**：`warning`

**触发条件**：目标 provider 不支持请求中指定的工具选择策略。

| 场景 | 说明 |
|------|------|
| `SpecificToolChoice("my_tool")` → 不支持指定工具的 provider | 降级为 `auto` |
| `RequiredToolChoice` → 不支持 `required` 的 provider | 降级为 `auto` |

---

## 结构化输出

### RESPONSE_FORMAT_DOWNGRADED

**等级**：`warning`

**触发条件**：目标 provider 不支持请求的结构化输出格式。

| 场景 | 说明 |
|------|------|
| `json_object` → Anthropic | Anthropic 不原生支持，建议在 system prompt 中说明 |
| `json_schema` → Anthropic | 同上 |
| `json_schema` → Ollama | 部分模型不支持 |

**处理策略**：在 system prompt / instructions 中明确描述期望的 JSON 格式，作为 prompt engineering 的替代方案。

---

## 有状态会话

### STATE_NOT_SUPPORTED

**等级**：`warning`

**触发条件**：请求包含有状态会话信息（`previous_response_id` / `conversation_id`），但目标 provider 不支持。

| 不支持的 Provider | 说明 |
|------------------|------|
| Anthropic | 无服务端状态 |
| Gemini | 无服务端状态 |
| Bedrock | 无服务端状态 |
| Ollama | 无服务端状态 |

**处理策略**：对于不支持有状态会话的 provider，需要客户端自行维护完整的对话历史（messages 列表）。

---

## 透传和兜底

### VENDOR_FIELD_PASSTHROUGH

**等级**：`info`

**触发条件**：遇到未知的 provider 专有字段或 content part，原样透传保留。

| 场景 | 说明 |
|------|------|
| 未知的 OpenAI content part type | 包装为 `ProviderBlock` |
| vendor 字段中的自定义数据 | 原样传递 |

**处理策略**：通常不需要处理。如果需要，可以在拦截器中清理这些字段。

### UNSUPPORTED_CONTENT_BLOCK

**等级**：`warning`

**触发条件**：遇到完全无法识别的内容块类型（通常是 `ProviderBlock`），降级为占位文本。

---

## 内置工具

### BUILTIN_TOOL_NOT_SUPPORTED

**等级**：`warning`

**触发条件**：请求中包含 provider 内置工具（`web_search`、`code_interpreter` 等），但目标 provider 不支持。

**处理策略**：内置工具是 L2 能力，仅 OpenAI Responses / Assistants API 支持。

---

## 流式传输

### STREAM_EVENT_DOWNGRADED

**等级**：`warning`

**触发条件**：流式传输事件无法准确映射到目标 provider 的事件格式。

---

## API 相关

### API_DEPRECATED_OR_MIGRATING

**等级**：`info`

**触发条件**：使用的 API 版本已弃用或正在迁移。

---

## Warning Code 速查表

| Code | 等级 | 说明 |
|------|------|------|
| `UNSUPPORTED_MODALITY` | warning | 不支持的内容模态 |
| `IMAGE_SOURCE_DOWNGRADED` | warning | 图片来源格式不兼容 |
| `FILE_REFERENCE_NOT_SUPPORTED` | warning | 不支持文件引用 |
| `ROLE_DOWNGRADED` | warning | 角色被合并/降级 |
| `UNSUPPORTED_ROLE` | warning | 角色不被支持 |
| `INVALID_TOOL_ARGUMENTS_JSON` | warning | 工具参数 JSON 解析失败 |
| `TOOL_CALL_BLOCK_DOWNGRADED` | warning | 工具调用块降级 |
| `TOOL_RESULT_BLOCK_DOWNGRADED` | warning | 工具结果块降级 |
| `TOOL_CHOICE_DOWNGRADED` | warning | 工具选择策略降级 |
| `RESPONSE_FORMAT_DOWNGRADED` | warning | 结构化输出格式降级 |
| `STATE_NOT_SUPPORTED` | warning | 不支持有状态会话 |
| `VENDOR_FIELD_PASSTHROUGH` | info | 厂商字段透传 |
| `UNSUPPORTED_CONTENT_BLOCK` | warning | 不支持的内容块 |
| `BUILTIN_TOOL_NOT_SUPPORTED` | warning | 不支持内置工具 |
| `STREAM_EVENT_DOWNGRADED` | warning | 流式事件降级 |
| `API_DEPRECATED_OR_MIGRATING` | info | API 版本弃用/迁移 |
| `UNSUPPORTED_CONTENT` | warning | 不支持的内容格式 |
