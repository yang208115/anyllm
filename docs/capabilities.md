# 能力矩阵

本文档详细说明 AnyLLM 的 Provider 能力分层体系和各 provider 的预设能力声明。

## 能力分层

AnyLLM 将 LLM provider 的能力分为三个层级：

| 层级 | 名称 | 说明 |
|------|------|------|
| **L0** | 基础聊天 | 所有 provider 都必须支持 |
| **L1** | 多模态 + 工具 | 大部分现代 provider 支持 |
| **L2** | 高级 Agent | 仅部分 provider 支持 |

### L0：基础聊天

| 能力 | 字段 | 说明 |
|------|------|------|
| 纯文本聊天 | `text` | 基础的文本对话 |
| 系统指令 | `system_instruction` | system prompt / instructions |
| 流式响应 | `streaming` | SSE / chunked transfer |

所有 provider 都应该支持 L0 能力。

### L1：多模态 + 工具调用

| 能力 | 字段 | 说明 |
|------|------|------|
| 图片输入 | `image_input` | user 消息中的 ImageBlock |
| 音频输入 | `audio_input` | user 消息中的 AudioBlock |
| 文件输入 | `file_input` | user 消息中的 FileBlock |
| 工具调用 | `tools` | function calling / tool use |
| 并行工具调用 | `parallel_tool_calls` | 单次响应多个 tool_calls |
| Tool Result Blocks | `tool_result_blocks` | ToolResult 作为 content block |
| JSON 对象输出 | `json_object` | response_format: json_object |
| JSON Schema 输出 | `json_schema` | Structured Outputs |

### L2：高级 Agent 特性

| 能力 | 字段 | 说明 |
|------|------|------|
| 有状态会话 | `stateful` | 服务端保留对话历史 |
| Previous Response ID | `previous_response_id` | 继续上一轮的响应 |
| 内置工具 | `builtin_tools` | web_search, file_search 等 |
| Developer 角色 | `developer_role` | developer 角色消息 |

## ProviderCapabilities 类

**模块**：`anyllm.adapters.base`

```python
class ProviderCapabilities(BaseModel):
    # L0
    text: bool = True
    system_instruction: bool = True
    streaming: bool = True
    # L1
    image_input: bool = False
    audio_input: bool = False
    file_input: bool = False
    tools: bool = False
    parallel_tool_calls: bool = False
    tool_result_blocks: bool = False
    json_object: bool = False
    json_schema: bool = False
    # L2
    stateful: bool = False
    previous_response_id: bool = False
    builtin_tools: bool = False
    developer_role: bool = False
```

默认值采用最保守的配置（仅 L0 能力），避免误报。适配器子类应覆盖 `capabilities` 属性以反映真实能力。

## 预设能力矩阵

**模块**：`anyllm.capabilities.matrix`

### 完整对照表

| 能力 | OpenAI Chat | OpenAI Resp | OpenAI Asst | Anthropic | Gemini | Bedrock | Ollama | Cloudflare |
|------|:-----------:|:-----------:|:-----------:|:---------:|:------:|:-------:|:------:|:----------:|
| **L0** | | | | | | | | |
| text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| system_instruction | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| streaming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **L1** | | | | | | | | |
| image_input | ✓ | ✓ | ✓ | ✓¹ | ✓ | ✓² | ✗³ | ✓³ |
| audio_input | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| file_input | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| tools | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓³ |
| parallel_tool_calls | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| tool_result_blocks | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| json_object | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓³ | ✓³ |
| json_schema | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓³ |
| **L2** | | | | | | | | |
| stateful | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| previous_response_id | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| builtin_tools | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| developer_role | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**注释**：
1. Anthropic 图片仅支持 base64，需 `ImageResolutionInterceptor` 转换
2. Bedrock 图片使用 bytes 格式（`image.source.bytes`）
3. Ollama / Cloudflare 的能力取决于具体部署的模型，表中为保守估计

### 各 Provider 详解

#### OpenAI Chat Completions

- **API 端点**：`/v1/chat/completions`
- **图片格式**：URL 和 base64 data URI 均支持
- **工具调用**：`message.tool_calls[]` 顶层列表，arguments 为 JSON 字符串
- **工具结果**：独立的 `tool` 角色消息，非 content block
- **JSON 模式**：支持 `json_object` 和 `json_schema`（Structured Outputs）
- **max_tokens**：可选

#### OpenAI Responses API

- **API 端点**：`/v1/responses`
- **有状态会话**：支持 `previous_response_id`
- **内置工具**：`web_search_preview`、`file_search`、`code_interpreter`
- **Developer 角色**：支持

#### Anthropic Messages API

- **API 端点**：`/v1/messages`
- **图片格式**：仅 base64（`{"type": "image", "source": {"type": "base64", ...}}`）
- **工具调用**：`tool_use` content block，嵌入 assistant 消息的 content 中
- **工具结果**：`tool_result` content block，必须在 user 消息的 content 中
- **max_tokens**：必填（默认填 1024）
- **消息规则**：user/assistant 必须交替出现，不允许连续同角色

#### Google Gemini

- **角色映射**：user / model（无 assistant / tool 角色）
- **工具调用**：`functionCall` 作为 Part
- **工具结果**：`functionResponse` 作为 Part
- **多模态**：支持图片、音频、文件

#### Amazon Bedrock Converse

- **图片格式**：bytes 格式（`image.source.bytes`）
- **工具调用**：`toolUse` content block
- **工具结果**：`toolResult` content block

#### Ollama (OpenAI-compatible)

- **兼容模式**：遵循 OpenAI Chat Completions API 格式
- **能力受限**：具体取决于部署的模型
- **默认端点**：`http://localhost:11434`

## 能力检查

适配器通过 `capabilities` 属性声明能力：

```python
adapter = OpenAIChatAdapter()
caps = adapter.capabilities

if not caps.image_input:
    print("此 provider 不支持图片输入")

if not caps.tools:
    print("此 provider 不支持工具调用")

if caps.stateful:
    print("支持有状态会话")
```

### 在代码中使用预设

```python
from anyllm.capabilities.matrix import (
    OPENAI_CHAT_CAPABILITIES,
    ANTHROPIC_CAPABILITIES,
    GEMINI_CAPABILITIES,
    BEDROCK_CONVERSE_CAPABILITIES,
    OLLAMA_OPENAI_COMPAT_CAPABILITIES,
    CLOUDFLARE_WORKERS_AI_CAPABILITIES,
)

# 比较两个 provider 的能力差异
openai_caps = OPENAI_CHAT_CAPABILITIES
anthropic_caps = ANTHROPIC_CAPABILITIES

# 找出 OpenAI 支持但 Anthropic 不支持的能力
for field in openai_caps.model_fields:
    openai_val = getattr(openai_caps, field)
    anthropic_val = getattr(anthropic_caps, field)
    if openai_val and not anthropic_val:
        print(f"Anthropic 不支持: {field}")
```

## 自定义能力声明

编写新适配器时，应如实声明能力：

```python
class MyProviderAdapter(BaseAdapter):
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            text=True,
            system_instruction=True,
            streaming=True,
            image_input=True,
            tools=True,
            json_object=True,
        )
```

如果 provider 的能力取决于具体模型（如 Ollama），可以在运行时动态返回：

```python
class OllamaAdapter(BaseAdapter):
    def __init__(self, model_capabilities=None):
        self._caps = model_capabilities or ProviderCapabilities()

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._caps
```
