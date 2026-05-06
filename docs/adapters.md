# 适配器开发指南

本文档介绍如何为新的 LLM provider 编写适配器，以及现有适配器的实现细节。

## 概述

适配器（Adapter）是 AnyLLM 的核心扩展点，负责 provider 原始 API 格式与 UIR 之间的双向转换。每个适配器需要实现 4 个方法：

```
Provider Request dict  ──→  request_to_uir()  ──→  UniversalRequest
Provider Response dict ──→  response_to_uir() ──→  UniversalResponse
UniversalRequest       ──→  uir_to_request()  ──→  Provider Request dict
UniversalResponse      ──→  uir_to_response() ──→  Provider Response dict
```

## 编写新适配器

### 第一步：继承 BaseAdapter

```python
from typing import Any, Dict
from anyllm.adapters.base import BaseAdapter, ProviderCapabilities
from anyllm.schema.request import UniversalRequest
from anyllm.schema.response import UniversalResponse
from anyllm.schema.warnings import ConversionResult, ConversionWarning


class MyProviderAdapter(BaseAdapter):
    """我的 provider 适配器。"""

    @property
    def provider_name(self) -> str:
        return "my_provider"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            text=True,
            system_instruction=True,
            streaming=True,
            image_input=True,
            tools=True,
        )

    def request_to_uir(
        self, raw_request: Dict[str, Any]
    ) -> ConversionResult[UniversalRequest]:
        ...

    def response_to_uir(
        self, raw_response: Dict[str, Any]
    ) -> ConversionResult[UniversalResponse]:
        ...

    def uir_to_request(
        self, request: UniversalRequest
    ) -> ConversionResult[Dict[str, Any]]:
        ...

    def uir_to_response(
        self, response: UniversalResponse
    ) -> ConversionResult[Dict[str, Any]]:
        ...
```

### 第二步：声明能力

覆盖 `capabilities` 属性，如实声明 provider 支持的功能：

```python
@property
def capabilities(self) -> ProviderCapabilities:
    return ProviderCapabilities(
        # L0: 基础能力
        text=True,
        system_instruction=True,
        streaming=True,
        # L1: 多模态 + 工具
        image_input=True,           # 是否支持图片输入
        audio_input=False,          # 是否支持音频输入
        file_input=False,           # 是否支持文件引用
        tools=True,                 # 是否支持 function calling
        parallel_tool_calls=True,   # 是否支持并行工具调用
        tool_result_blocks=True,    # tool_result 是否作为 content block
        json_object=False,          # 是否支持 json_object 输出
        json_schema=False,          # 是否支持 Structured Outputs
        # L2: 高级能力
        stateful=False,
        previous_response_id=False,
        builtin_tools=False,
        developer_role=False,
    )
```

### 第三步：实现 request_to_uir

将 provider 原始请求 dict 转换为 UIR。这是「入站翻译」，需要处理：

1. **提取模型信息** → `ModelRef`
2. **解析消息** → `List[Message]`，包含 content blocks
3. **处理 system prompt** → 可能在消息列表中，也可能在单独字段
4. **解析工具定义** → `List[ToolDef]`
5. **解析工具调用** → `ToolCall`（JSON string → dict）
6. **处理多模态内容** → `ImageBlock`、`AudioBlock` 等
7. **透传未知字段** → `ProviderBlock` + warning

```python
def request_to_uir(
    self, raw_request: Dict[str, Any]
) -> ConversionResult[UniversalRequest]:
    warnings: list[ConversionWarning] = []

    # 1. 提取模型
    model = ModelRef(
        provider=self.provider_name,
        name=raw_request.get("model", "unknown"),
    )

    # 2. 解析消息
    messages = []
    instructions = []
    for msg_dict in raw_request.get("messages", []):
        role = msg_dict.get("role")
        if role == "system":
            instructions.append(TextBlock(text=msg_dict.get("content", "")))
            continue
        messages.append(self._parse_message(msg_dict, warnings))

    # 3. 解析生成配置
    generation = GenerationConfig(
        temperature=raw_request.get("temperature"),
        max_output_tokens=raw_request.get("max_tokens"),
    )

    request = UniversalRequest(
        model=model,
        messages=messages,
        instructions=instructions,
        generation=generation,
    )

    return ConversionResult(value=request, warnings=warnings)
```

### 第四步：实现 uir_to_request

将 UIR 转换为 provider 原始请求 dict。这是「出站翻译」，是最复杂的部分，需要处理：

1. **system prompt 位置** — 不同 provider 放置位置不同
2. **content block 格式** — 文本、图片、工具调用的表达方式
3. **字段名映射** — `max_tokens` vs `maxTokens` vs `max_output_tokens`
4. **不兼容特性** — 发出 warning 而非抛异常

```python
def uir_to_request(
    self, request: UniversalRequest
) -> ConversionResult[Dict[str, Any]]:
    warnings: list[ConversionWarning] = []

    raw: Dict[str, Any] = {
        "model": request.model.name,
        "messages": [],
    }

    # 处理 system prompt
    if request.instructions:
        raw["system"] = blocks_to_plain_text(
            request.instructions, warnings, "instructions"
        )

    # 转换消息
    for idx, msg in enumerate(request.messages):
        raw["messages"].append(
            self._convert_message(msg, warnings, f"messages[{idx}]")
        )

    # 转换生成配置
    if request.generation.temperature is not None:
        raw["temperature"] = request.generation.temperature

    # 处理 vendor 透传
    vendor_data = request.vendor.get(self.provider_name, {})
    if vendor_data:
        raw.update(vendor_data)

    return ConversionResult(value=raw, warnings=warnings)
```

### 第五步：实现 response_to_uir 和 uir_to_response

```python
def response_to_uir(
    self, raw_response: Dict[str, Any]
) -> ConversionResult[UniversalResponse]:
    warnings: list[ConversionWarning] = []

    response = UniversalResponse(
        id=raw_response.get("id"),
        model=raw_response.get("model"),
        output=[...],  # 解析输出消息
        stop_reason=normalize_stop_reason(
            self.provider_name,
            raw_response.get("stop_reason")
        ),
        usage=self._parse_usage(raw_response),
        raw=raw_response,
    )

    return ConversionResult(value=response, warnings=warnings)
```

## 实现规范

### 规范一：始终返回 ConversionResult

所有转换方法必须返回 `ConversionResult[T]`，即使没有 warning：

```python
# 正确
return ConversionResult(value=result_dict, warnings=[])

# 错误 — 不要直接返回 dict
return result_dict
```

### 规范二：降级而非报错

遇到不支持的特性时，发出 warning 并尽量降级，而非抛出异常：

```python
# 正确 — 降级 + warning
if isinstance(block, AudioBlock):
    warnings.append(ConversionWarning(
        code="UNSUPPORTED_MODALITY",
        path=f"messages[{idx}].content[{j}]",
        message="该 provider 不支持音频输入，已忽略。",
    ))
    continue

# 错误 — 直接抛异常
if isinstance(block, AudioBlock):
    raise ValueError("不支持音频")
```

只有当数据格式严重损坏导致无法继续转换时，才应该抛出异常。

### 规范三：透传而非丢弃

遇到无法识别的字段或 content part 时，包装为 `ProviderBlock` 透传：

```python
# 无法识别的 content part
provider_block = ProviderBlock(
    provider=self.provider_name,
    value=unknown_part,
)
blocks.append(provider_block)
warnings.append(ConversionWarning(
    code="VENDOR_FIELD_PASSTHROUGH",
    path=block_path,
    message=f"未知内容类型，已透传保留。",
    severity="info",
))
```

### 规范四：使用 lowering 工具函数

`anyllm.conversion.lowering` 提供了通用的降级函数，适配器应复用而非重写：

| 函数 | 用途 |
|------|------|
| `blocks_to_plain_text(blocks, warnings, path)` | ContentBlock 列表 → 纯文本 |
| `tool_result_content_to_text(result, warnings, path)` | ToolResult 内容 → 纯文本 |
| `extract_text_from_blocks(blocks)` | 仅提取 TextBlock 文本（无 warning） |
| `serialize_tool_arguments(arguments)` | dict → JSON 字符串 |

## 现有适配器

### OpenAIChatAdapter

**模块**：`anyllm.adapters.openai_chat`

适配 OpenAI Chat Completions API (`/v1/chat/completions`)。

**关键映射规则**：

| UIR 结构 | OpenAI 格式 |
|----------|-------------|
| `instructions` | 首条 `{"role": "system", "content": str}` |
| `ImageBlock(url)` | `{"type": "image_url", "image_url": {"url": ...}}` |
| `ImageBlock(base64)` | `{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}` |
| `ToolCall` | `message.tool_calls[{"id", "function": {"name", "arguments": str}}]` |
| `ToolResult` | `{"role": "tool", "tool_call_id": ..., "content": str}` |
| `generation.max_output_tokens` | `max_tokens` |
| `generation.stop` | `stop` |

**特殊处理**：
- `ToolCallBlock` 在 content 中 → 自动提升到 `message.tool_calls` 顶层
- `ToolResultBlock` 在 content 中 → 自动转为独立的 `tool` 角色消息
- `tool_call.arguments` 从 dict → JSON 字符串（OpenAI 要求字符串格式）

### AnthropicAdapter

**模块**：`anyllm.adapters.anthropic`

适配 Anthropic Messages API (`/v1/messages`)。

**关键映射规则**：

| UIR 结构 | Anthropic 格式 |
|----------|---------------|
| `instructions` | 顶层 `system` 字段（纯文本） |
| `system/developer` 角色消息 | 合并到顶层 `system` + `ROLE_DOWNGRADED` warning |
| `ImageBlock(base64)` | `{"type": "image", "source": {"type": "base64", ...}}` |
| `ImageBlock(url)` | `IMAGE_SOURCE_DOWNGRADED` warning（需先经 ImageResolutionInterceptor） |
| `ToolCall` | content 中的 `{"type": "tool_use", ...}` |
| `ToolResult` | content 中的 `{"type": "tool_result", ...}` |
| `generation.max_output_tokens` | `max_tokens`（必填，默认 1024） |
| `response_format` (json_*) | `RESPONSE_FORMAT_DOWNGRADED` warning |

**Anthropic 特有约束**：
- `max_tokens` 是必填字段，适配器默认填 1024
- 不支持 `system` / `developer` 角色消息出现在 messages 中
- 不支持 URL 图片（需要 `ImageResolutionInterceptor` 预转换为 base64）
- `tool_result` 必须出现在 `user` 角色消息的 content 中（需要 `RoleConsolidationInterceptor`）
- 不支持 `json_object` / `json_schema` 响应格式
- 不支持有状态会话

### GeminiAdapter

**模块**：`anyllm.adapters.gemini`

适配 Google Gemini generateContent API (`/v1beta/models/{model}:generateContent`)。

**关键映射规则**：

| UIR 结构 | Gemini 格式 |
|----------|-------------|
| `instructions` | `systemInstruction.parts` |
| `Message(role="assistant")` | `contents[].role = "model"` |
| `ToolCall` | `parts[].functionCall` |
| `ToolResult` | `parts[].functionResponse` |
| `generation.max_output_tokens` | `generationConfig.maxOutputTokens` |
| `response_format=json_object` | `generationConfig.responseMimeType=application/json` |
| `response_format=json_schema` | `generationConfig.responseSchema` |

**Gemini 特有约束**：
- provider 名建议统一注册为 `google`（`gemini` 作为网关 alias 可用）
- endpoint model 使用 URL path 传递
- `generateContent` 仅支持 function tools，其他 tool type 会发 warning 并忽略
- 不支持 UIR `state` 会话字段（会发 `STATE_NOT_SUPPORTED` warning）

## 注册适配器

将适配器注册到转换器或网关：

```python
# 注册到 UniversalConverter
converter = UniversalConverter()
converter.register_adapter("my_provider", MyProviderAdapter())

# 注册到 AnyLLMGateway
gateway = AnyLLMGateway()
gateway.register_provider("my_provider", ProviderConfig(
    adapter=MyProviderAdapter(),
    api_base="https://api.my-provider.com",
    api_key="xxx",
))
```

## 测试适配器

建议为每个适配器编写以下测试用例：

```python
import asyncio

def test_roundtrip():
    """验证 request → UIR → request 的往返一致性。"""
    adapter = MyProviderAdapter()

    original = {
        "model": "my-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # 正向：provider dict → UIR
    uir_result = adapter.request_to_uir(original)
    assert not uir_result.has_errors
    assert len(uir_result.value.messages) == 1

    # 反向：UIR → provider dict
    back_result = adapter.uir_to_request(uir_result.value)
    assert back_result.value["model"] == "my-model"


def test_multimodal():
    """验证多模态内容的转换。"""
    ...


def test_tool_calling():
    """验证工具调用的转换。"""
    ...


def test_unsupported_features():
    """验证不支持的特性产生正确的 warning。"""
    ...
```
