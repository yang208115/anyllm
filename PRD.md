# PRD：Universal AI Message IR 通用中间格式

## 1. 背景

我们要支持这些 API 之间互转：

* Anthropic Messages API
* Google Gemini API
* Amazon Bedrock Converse / model APIs
* Ollama API / OpenAI-compatible API
* OpenAI Assistants API
* OpenAI Responses API
* Cloudflare Workers AI
* OpenAI Chat Completions API

这些 API 最大的问题不是字段名不同，而是**抽象层级不同**：

* Chat Completions / Ollama / Workers AI：偏简单 chat message。
* Anthropic / Gemini / Bedrock：偏 content block / part。
* OpenAI Responses：偏 response item / stateful workflow。
* OpenAI Assistants：偏 assistant + thread + run 的 agent 抽象，且官方已进入向 Responses API 迁移的路径，Assistants API 计划于 2026-08-26 关闭。([OpenAI开发者][1])

因此不能选某一家 API 当内部格式。正确做法是设计一个 **Universal Intermediate Representation，简称 UIR**。

---

# 2. 目标

## 2.1 核心目标

设计一个通用中间格式，使不同模型 API 可以通过：

```txt
Provider Request -> UIR -> Provider Request
Provider Response -> UIR -> Provider Response
```

实现互转。

## 2.2 设计原则

### 原则一：信息最大保真

不要为了兼容低阶接口，把高阶接口能力提前丢掉。

例如 OpenAI Responses 支持 stateful interactions、built-in tools、function calling；Chat Completions 则需要用户手动维护 conversation state。([OpenAI开发者][2])

所以 UIR 需要保留：

```txt
messages
content blocks
tool calls
tool results
state
vendor-specific fields
usage
reasoning / thinking
structured output
stream events
```

### 原则二：降级可见

不是所有能力都能无损转换。

比如：

```txt
OpenAI Responses previous_response_id -> Ollama OpenAI-compatible Responses
```

Ollama 官方说明其 OpenAI-compatible Responses 只支持非状态模式，不支持 `previous_response_id` 或 conversation。([Ollama 文档][3])

因此转换结果必须带 warnings。

### 原则三：content 永远是 block array

Anthropic、Gemini、Bedrock 都天然接近 block / part 模型。Anthropic Messages API 支持 content block 和 tool use；Gemini function calling 用 function calls / responses；Bedrock Converse 也是 messages + content + tool use 的统一接口。([Claude 平台][4])

所以内部不要用：

```python
content: str
```

而要用：

```python
content: list[ContentBlock]
```

---

# 3. 非目标

第一版不解决：

```txt
1. 完全统一所有 provider 的计费字段
2. 完全统一所有 provider 的安全过滤原因
3. 完全统一 realtime / live audio session
4. 完全统一 agent runtime 调度
5. 完全统一文件存储系统
6. 完全统一 provider 内置工具行为
```

这些都可以通过 `vendor` 字段保留。

---

# 4. 总体架构

```txt
                  ┌─────────────────────┐
                  │ Provider A Request   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  A -> UIR Adapter    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Universal AI IR      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  UIR -> B Adapter    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Provider B Request   │
                  └─────────────────────┘
```

每个 provider 至少实现两个适配器：

```python
class ProviderAdapter:
    def request_to_uir(self, raw_request: dict) -> "ConversionResult[UniversalRequest]":
        ...

    def response_to_uir(self, raw_response: dict) -> "ConversionResult[UniversalResponse]":
        ...

    def uir_to_request(self, request: "UniversalRequest") -> "ConversionResult[dict]":
        ...

    def uir_to_response(self, response: "UniversalResponse") -> "ConversionResult[dict]":
        ...
```

---

# 5. 核心数据模型

## 5.1 UniversalRequest

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union


ProviderName = Literal[
    "openai",
    "anthropic",
    "google",
    "bedrock",
    "ollama",
    "cloudflare",
    "unknown",
]


Role = Literal[
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
]


StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_calls",
    "content_filter",
    "error",
    "unknown",
]


@dataclass
class ModelRef:
    name: str
    provider: ProviderName = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    conversation_id: Optional[str] = None
    thread_id: Optional[str] = None
    run_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    provider_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalRequest:
    version: str
    model: ModelRef
    messages: list["Message"]

    instructions: list["ContentBlock"] = field(default_factory=list)

    tools: list["ToolDef"] = field(default_factory=list)
    tool_choice: Optional["ToolChoice"] = None

    response_format: Optional["ResponseFormat"] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    stream: bool = False
    state: ConversationState = field(default_factory=ConversationState)

    metadata: dict[str, Any] = field(default_factory=dict)
    vendor: dict[str, Any] = field(default_factory=dict)
```

---

# 6. Message 设计

## 6.1 Message

```python
@dataclass
class Message:
    role: Role
    content: list["ContentBlock"]

    id: Optional[str] = None
    name: Optional[str] = None

    tool_calls: list["ToolCall"] = field(default_factory=list)
    tool_results: list["ToolResult"] = field(default_factory=list)

    provider: dict[str, Any] = field(default_factory=dict)
```

为什么 `tool_calls` 和 `tool_results` 同时存在？

因为不同 API 表达工具调用的位置不同：

```txt
OpenAI Chat Completions:
assistant.message.tool_calls
tool role message

Anthropic:
assistant content 里有 tool_use block
user content 里有 tool_result block

Gemini:
model part 里有 functionCall
user part 里有 functionResponse

Bedrock:
content 里有 toolUse / toolResult

Ollama:
OpenAI-compatible tool_calls
```

Ollama 官方也支持 tool calling。([Ollama 文档][5])

所以 UIR 同时支持：

```python
message.tool_calls
message.tool_results
content block: tool_call
content block: tool_result
```

转换时可以互相派生。

---

# 7. ContentBlock 设计

## 7.1 ContentBlock 类型

```python
ContentBlockType = Literal[
    "text",
    "image",
    "audio",
    "file",
    "thinking",
    "refusal",
    "tool_call",
    "tool_result",
    "provider_block",
]


@dataclass
class MediaSource:
    kind: Literal["url", "base64", "file_id", "bytes"]
    value: Any
    mime_type: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class TextBlock:
    type: Literal["text"]
    text: str
    annotations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ImageBlock:
    type: Literal["image"]
    source: MediaSource
    detail: Optional[Literal["low", "high", "auto"]] = None


@dataclass
class AudioBlock:
    type: Literal["audio"]
    source: MediaSource
    format: Optional[str] = None


@dataclass
class FileBlock:
    type: Literal["file"]
    source: MediaSource
    mime_type: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class ThinkingBlock:
    type: Literal["thinking"]
    text: Optional[str] = None
    encrypted: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class RefusalBlock:
    type: Literal["refusal"]
    text: str


@dataclass
class ToolCallBlock:
    type: Literal["tool_call"]
    call: "ToolCall"


@dataclass
class ToolResultBlock:
    type: Literal["tool_result"]
    result: "ToolResult"


@dataclass
class ProviderBlock:
    type: Literal["provider_block"]
    provider: str
    value: Any


ContentBlock = Union[
    TextBlock,
    ImageBlock,
    AudioBlock,
    FileBlock,
    ThinkingBlock,
    RefusalBlock,
    ToolCallBlock,
    ToolResultBlock,
    ProviderBlock,
]
```

---

# 8. Tool 设计

## 8.1 ToolDef

```python
@dataclass
class ToolDef:
    type: Literal["function", "builtin", "mcp", "provider"]
    name: str
    description: Optional[str] = None

    input_schema: Optional[dict[str, Any]] = None
    output_schema: Optional[dict[str, Any]] = None

    provider: dict[str, Any] = field(default_factory=dict)
```

第一版重点支持 function tool。

```python
weather_tool = ToolDef(
    type="function",
    name="get_weather",
    description="Get weather by city and date",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["city", "date"],
    },
)
```

## 8.2 ToolCall

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any

    raw_arguments: Optional[str] = None
    provider: dict[str, Any] = field(default_factory=dict)
```

`arguments` 必须尽量规范成 Python dict。

如果 provider 返回的是字符串 JSON：

```json
"{\"city\":\"Tokyo\"}"
```

转换成：

```python
arguments = {"city": "Tokyo"}
raw_arguments = "{\"city\":\"Tokyo\"}"
```

如果 JSON parse 失败：

```python
arguments = {}
raw_arguments = original_string
warnings.append("INVALID_TOOL_ARGUMENTS_JSON")
```

## 8.3 ToolResult

```python
@dataclass
class ToolResult:
    call_id: str
    content: list[ContentBlock]

    name: Optional[str] = None
    is_error: bool = False
    provider: dict[str, Any] = field(default_factory=dict)
```

---

# 9. Response Format 设计

结构化输出统一成三种：

```python
@dataclass
class TextResponseFormat:
    type: Literal["text"] = "text"


@dataclass
class JsonObjectResponseFormat:
    type: Literal["json_object"] = "json_object"


@dataclass
class JsonSchemaResponseFormat:
    type: Literal["json_schema"]
    name: str
    schema: dict[str, Any]
    strict: bool = False


ResponseFormat = Union[
    TextResponseFormat,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
]
```

OpenAI Structured Outputs 区分 tool schema 和 response_format schema：tool 适合模型调用函数，`response_format` 更适合约束模型直接返回结构化结果。([OpenAI开发者][6])

所以 UIR 必须同时保留：

```python
tools[].input_schema
response_format.schema
```

不能混为一谈。

---

# 10. ConversionResult 与 Warning

所有转换函数都不能只返回 dict，必须返回：

```python
@dataclass
class ConversionWarning:
    code: str
    path: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


@dataclass
class ConversionResult:
    value: Any
    warnings: list[ConversionWarning] = field(default_factory=list)

    def add_warning(
        self,
        code: str,
        path: str,
        message: str,
        severity: Literal["info", "warning", "error"] = "warning",
    ) -> None:
        self.warnings.append(
            ConversionWarning(
                code=code,
                path=path,
                message=message,
                severity=severity,
            )
        )
```

典型 warning：

```python
UNSUPPORTED_ROLE
UNSUPPORTED_MODALITY
STATE_NOT_SUPPORTED
TOOL_CHOICE_DOWNGRADED
RESPONSE_FORMAT_DOWNGRADED
VENDOR_FIELD_PASSTHROUGH
INVALID_TOOL_ARGUMENTS_JSON
STREAM_EVENT_DOWNGRADED
BUILTIN_TOOL_NOT_SUPPORTED
FILE_REFERENCE_NOT_SUPPORTED
```

---

# 11. 能力分层

建议把能力分三层。

## L0：纯文本聊天

```txt
system / user / assistant
text content
temperature
max_tokens
stream
```

所有 provider 都应该支持。

## L1：多模态 + 工具调用

```txt
content blocks
image input
tool definitions
tool calls
tool results
json schema output
usage
finish reason
```

Anthropic、Gemini、Bedrock、OpenAI、部分 Ollama/Workers AI 模型支持。

Cloudflare Workers AI 支持 OpenAI-compatible `/v1/chat/completions` 和 `/v1/embeddings`，具体高级能力还要看模型。([Cloudflare Docs][7])

## L2：高级 agent / provider 特性

```txt
stateful conversation
assistant/thread/run
previous_response_id
built-in tools
computer use
file search
web search
reasoning summaries
prompt cache
provider-specific safety settings
```

这层默认不保证跨 provider 无损。

---

# 12. Provider Capability Matrix

```python
@dataclass
class ProviderCapabilities:
    text: bool = True
    image_input: bool = False
    audio_input: bool = False
    file_input: bool = False

    tools: bool = False
    parallel_tool_calls: bool = False
    tool_result_blocks: bool = False

    json_object: bool = False
    json_schema: bool = False

    stateful: bool = False
    previous_response_id: bool = False

    builtin_tools: bool = False
    streaming: bool = True

    system_instruction: bool = True
    developer_role: bool = False
```

示例：

```python
OPENAI_RESPONSES_CAPS = ProviderCapabilities(
    text=True,
    image_input=True,
    tools=True,
    parallel_tool_calls=True,
    json_object=True,
    json_schema=True,
    stateful=True,
    previous_response_id=True,
    builtin_tools=True,
    streaming=True,
    developer_role=True,
)

OPENAI_CHAT_CAPS = ProviderCapabilities(
    text=True,
    image_input=True,
    tools=True,
    parallel_tool_calls=True,
    json_object=True,
    json_schema=True,
    stateful=False,
    previous_response_id=False,
    streaming=True,
    developer_role=True,
)

OLLAMA_OPENAI_COMPAT_CAPS = ProviderCapabilities(
    text=True,
    image_input=False,  # 取决于具体模型和端点
    tools=True,
    json_object=True,
    json_schema=False,  # 取决于模型/端点，默认保守
    stateful=False,
    previous_response_id=False,
    streaming=True,
)

CLOUDFLARE_WORKERS_AI_CAPS = ProviderCapabilities(
    text=True,
    image_input=True,   # 取决于模型
    tools=True,         # 取决于模型
    json_object=True,   # 取决于模型
    json_schema=True,   # 取决于模型
    stateful=False,
    streaming=True,
)
```

---

# 13. 转换流程

## 13.1 总入口

```python
class UniversalConverter:
    def __init__(self):
        self.adapters: dict[str, ProviderAdapter] = {}

    def register(self, provider: str, adapter: "ProviderAdapter") -> None:
        self.adapters[provider] = adapter

    def request_to_uir(
        self,
        provider: str,
        raw_request: dict,
    ) -> ConversionResult:
        adapter = self.adapters[provider]
        return adapter.request_to_uir(raw_request)

    def uir_to_request(
        self,
        provider: str,
        request: UniversalRequest,
    ) -> ConversionResult:
        adapter = self.adapters[provider]
        return adapter.uir_to_request(request)

    def convert_request(
        self,
        source_provider: str,
        target_provider: str,
        raw_request: dict,
    ) -> ConversionResult:
        source_result = self.request_to_uir(source_provider, raw_request)
        target_result = self.uir_to_request(target_provider, source_result.value)

        return ConversionResult(
            value=target_result.value,
            warnings=source_result.warnings + target_result.warnings,
        )
```

---

# 14. 通用工具函数

## 14.1 文本 block 拼接

某些 provider 只接受 string content，需要降级。

```python
def blocks_to_plain_text(
    blocks: list[ContentBlock],
    warnings: list[ConversionWarning],
    path: str,
) -> str:
    parts: list[str] = []

    for i, block in enumerate(blocks):
        block_path = f"{path}[{i}]"

        if isinstance(block, TextBlock):
            parts.append(block.text)

        elif isinstance(block, ImageBlock):
            parts.append("[Image omitted]")
            warnings.append(
                ConversionWarning(
                    code="UNSUPPORTED_MODALITY",
                    path=block_path,
                    message="Target provider does not support image block here.",
                )
            )

        elif isinstance(block, FileBlock):
            parts.append(f"[File omitted: {block.filename or 'unknown'}]")
            warnings.append(
                ConversionWarning(
                    code="FILE_REFERENCE_NOT_SUPPORTED",
                    path=block_path,
                    message="Target provider does not support file block here.",
                )
            )

        elif isinstance(block, ToolCallBlock):
            parts.append(f"[Tool call omitted: {block.call.name}]")
            warnings.append(
                ConversionWarning(
                    code="TOOL_CALL_BLOCK_DOWNGRADED",
                    path=block_path,
                    message="Tool call block was converted to plain text placeholder.",
                )
            )

        elif isinstance(block, ToolResultBlock):
            parts.append("[Tool result omitted]")
            warnings.append(
                ConversionWarning(
                    code="TOOL_RESULT_BLOCK_DOWNGRADED",
                    path=block_path,
                    message="Tool result block was converted to plain text placeholder.",
                )
            )

        else:
            parts.append("[Unsupported content omitted]")
            warnings.append(
                ConversionWarning(
                    code="UNSUPPORTED_CONTENT_BLOCK",
                    path=block_path,
                    message=f"Unsupported block type: {getattr(block, 'type', 'unknown')}",
                )
            )

    return "\n".join(parts)
```

## 14.2 JSON 参数解析

```python
import json


def parse_tool_arguments(value: Any) -> tuple[Any, Optional[str], Optional[ConversionWarning]]:
    if isinstance(value, dict):
        return value, None, None

    if isinstance(value, str):
        try:
            return json.loads(value), value, None
        except json.JSONDecodeError:
            return {}, value, ConversionWarning(
                code="INVALID_TOOL_ARGUMENTS_JSON",
                path="tool_call.arguments",
                message="Tool call arguments are not valid JSON.",
                severity="warning",
            )

    return value, None, None
```

---

# 15. OpenAI Chat Completions Adapter

## 15.1 OpenAI Chat -> UIR

OpenAI Chat Completions 典型格式：

```python
{
    "model": "gpt-4.1",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"}
    ],
    "tools": [...],
    "tool_choice": "auto",
}
```

转换：

```python
class OpenAIChatAdapter:
    provider = "openai_chat"

    def request_to_uir(self, raw: dict) -> ConversionResult:
        warnings: list[ConversionWarning] = []

        messages = []
        instructions = []

        for idx, msg in enumerate(raw.get("messages", [])):
            role = msg.get("role")
            content = self._parse_content(msg.get("content"), warnings, f"messages[{idx}].content")

            if role == "system":
                instructions.extend(content)
                continue

            tool_calls = []
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                args, raw_args, warning = parse_tool_arguments(fn.get("arguments", {}))
                if warning:
                    warnings.append(warning)

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=args,
                        raw_arguments=raw_args,
                        provider={"raw": tc},
                    )
                )

            messages.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    provider={"raw": msg},
                )
            )

        uir = UniversalRequest(
            version="uai.v1",
            model=ModelRef(
                provider="openai",
                name=raw["model"],
            ),
            instructions=instructions,
            messages=messages,
            tools=[self._parse_tool(t) for t in raw.get("tools", [])],
            tool_choice=self._parse_tool_choice(raw.get("tool_choice")),
            response_format=self._parse_response_format(raw.get("response_format")),
            generation=GenerationConfig(
                temperature=raw.get("temperature"),
                top_p=raw.get("top_p"),
                max_output_tokens=raw.get("max_tokens"),
                stop=raw.get("stop"),
                presence_penalty=raw.get("presence_penalty"),
                frequency_penalty=raw.get("frequency_penalty"),
            ),
            stream=bool(raw.get("stream", False)),
            vendor={"openai_chat": self._extract_unknown_fields(raw)},
        )

        return ConversionResult(value=uir, warnings=warnings)

    def _parse_content(
        self,
        content: Any,
        warnings: list[ConversionWarning],
        path: str,
    ) -> list[ContentBlock]:
        if content is None:
            return []

        if isinstance(content, str):
            return [TextBlock(type="text", text=content)]

        if isinstance(content, list):
            blocks = []
            for i, part in enumerate(content):
                part_type = part.get("type")

                if part_type == "text":
                    blocks.append(TextBlock(type="text", text=part.get("text", "")))

                elif part_type in ("image_url", "input_image"):
                    url = part.get("image_url", {}).get("url") or part.get("image_url")
                    blocks.append(
                        ImageBlock(
                            type="image",
                            source=MediaSource(kind="url", value=url),
                            detail=part.get("detail"),
                        )
                    )

                else:
                    blocks.append(
                        ProviderBlock(
                            type="provider_block",
                            provider="openai",
                            value=part,
                        )
                    )
                    warnings.append(
                        ConversionWarning(
                            code="VENDOR_FIELD_PASSTHROUGH",
                            path=f"{path}[{i}]",
                            message=f"Unknown OpenAI content part type: {part_type}",
                            severity="info",
                        )
                    )

            return blocks

        warnings.append(
            ConversionWarning(
                code="UNSUPPORTED_CONTENT",
                path=path,
                message="Unsupported OpenAI content format.",
            )
        )
        return [TextBlock(type="text", text=str(content))]
```

## 15.2 UIR -> OpenAI Chat

```python
    def uir_to_request(self, request: UniversalRequest) -> ConversionResult:
        warnings: list[ConversionWarning] = []

        messages = []

        if request.instructions:
            messages.append({
                "role": "system",
                "content": blocks_to_plain_text(
                    request.instructions,
                    warnings,
                    "instructions",
                ),
            })

        for idx, msg in enumerate(request.messages):
            out = {
                "role": self._map_role(msg.role, warnings, f"messages[{idx}].role"),
                "content": self._blocks_to_openai_content(
                    msg.content,
                    warnings,
                    f"messages[{idx}].content",
                ),
            }

            if msg.name:
                out["name"] = msg.name

            if msg.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in msg.tool_calls
                ]

            if msg.role == "tool" and msg.tool_results:
                result = msg.tool_results[0]
                out["tool_call_id"] = result.call_id
                out["content"] = blocks_to_plain_text(
                    result.content,
                    warnings,
                    f"messages[{idx}].tool_results[0].content",
                )

            messages.append(out)

        raw = {
            "model": request.model.name,
            "messages": messages,
        }

        if request.tools:
            raw["tools"] = [self._tool_to_openai(t) for t in request.tools]

        if request.tool_choice:
            raw["tool_choice"] = self._tool_choice_to_openai(request.tool_choice, warnings)

        if request.response_format:
            raw["response_format"] = self._response_format_to_openai(
                request.response_format,
                warnings,
            )

        self._apply_generation(raw, request.generation)
        raw["stream"] = request.stream

        if request.state.previous_response_id or request.state.conversation_id:
            warnings.append(
                ConversionWarning(
                    code="STATE_NOT_SUPPORTED",
                    path="state",
                    message="OpenAI Chat Completions does not support previous_response_id or conversation state directly.",
                )
            )

        return ConversionResult(value=raw, warnings=warnings)
```

---

# 16. OpenAI Responses Adapter

OpenAI Responses 是富格式目标。官方文档描述它支持 text/image input、stateful interactions、built-in tools、function calling 等。([OpenAI开发者][2])

## 16.1 UIR -> Responses

```python
class OpenAIResponsesAdapter:
    provider = "openai_responses"

    def uir_to_request(self, request: UniversalRequest) -> ConversionResult:
        warnings: list[ConversionWarning] = []

        raw: dict[str, Any] = {
            "model": request.model.name,
            "input": [],
        }

        if request.instructions:
            raw["instructions"] = blocks_to_plain_text(
                request.instructions,
                warnings,
                "instructions",
            )

        for idx, msg in enumerate(request.messages):
            raw["input"].append(
                self._message_to_response_item(
                    msg,
                    warnings,
                    f"messages[{idx}]",
                )
            )

        if request.tools:
            raw["tools"] = [self._tool_to_responses(t, warnings) for t in request.tools]

        if request.tool_choice:
            raw["tool_choice"] = self._tool_choice_to_responses(
                request.tool_choice,
                warnings,
            )

        if request.response_format:
            raw["text"] = {
                "format": self._response_format_to_responses(
                    request.response_format,
                    warnings,
                )
            }

        if request.state.previous_response_id:
            raw["previous_response_id"] = request.state.previous_response_id

        if request.state.conversation_id:
            raw["conversation"] = request.state.conversation_id

        self._apply_generation(raw, request.generation)

        if request.stream:
            raw["stream"] = True

        return ConversionResult(value=raw, warnings=warnings)

    def _message_to_response_item(
        self,
        msg: Message,
        warnings: list[ConversionWarning],
        path: str,
    ) -> dict:
        item = {
            "role": self._map_role(msg.role, warnings, f"{path}.role"),
            "content": [],
        }

        for i, block in enumerate(msg.content):
            item["content"].append(
                self._block_to_responses_content(
                    block,
                    warnings,
                    f"{path}.content[{i}]",
                )
            )

        for call in msg.tool_calls:
            item["content"].append({
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            })

        for result in msg.tool_results:
            item["content"].append({
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": blocks_to_plain_text(
                    result.content,
                    warnings,
                    f"{path}.tool_results",
                ),
            })

        return item
```

---

# 17. Anthropic Adapter

Anthropic Messages API 是典型 content block 模型。工具调用一般是：

```txt
assistant -> tool_use
user -> tool_result
```

Anthropic 文档也明确有 `tool_result` content blocks。([Claude 平台][8])

## 17.1 UIR -> Anthropic

```python
class AnthropicMessagesAdapter:
    provider = "anthropic"

    def uir_to_request(self, request: UniversalRequest) -> ConversionResult:
        warnings: list[ConversionWarning] = []

        raw = {
            "model": request.model.name,
            "messages": [],
            "max_tokens": request.generation.max_output_tokens or 1024,
        }

        if request.instructions:
            raw["system"] = blocks_to_plain_text(
                request.instructions,
                warnings,
                "instructions",
            )

        for idx, msg in enumerate(request.messages):
            if msg.role in ("system", "developer"):
                raw["system"] = (
                    raw.get("system", "")
                    + "\n"
                    + blocks_to_plain_text(msg.content, warnings, f"messages[{idx}].content")
                ).strip()
                warnings.append(
                    ConversionWarning(
                        code="ROLE_DOWNGRADED",
                        path=f"messages[{idx}].role",
                        message="Anthropic uses top-level system instead of system/developer messages.",
                    )
                )
                continue

            anthropic_msg = {
                "role": self._map_role(msg.role, warnings, f"messages[{idx}].role"),
                "content": [],
            }

            for j, block in enumerate(msg.content):
                anthropic_msg["content"].append(
                    self._block_to_anthropic(
                        block,
                        warnings,
                        f"messages[{idx}].content[{j}]",
                    )
                )

            for call in msg.tool_calls:
                anthropic_msg["content"].append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                })

            for result in msg.tool_results:
                anthropic_msg["content"].append({
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": blocks_to_plain_text(
                        result.content,
                        warnings,
                        f"messages[{idx}].tool_results",
                    ),
                    "is_error": result.is_error,
                })

            raw["messages"].append(anthropic_msg)

        if request.tools:
            raw["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.input_schema or {
                        "type": "object",
                        "properties": {},
                    },
                }
                for tool in request.tools
                if tool.type == "function"
            ]

        if request.tool_choice:
            raw["tool_choice"] = self._tool_choice_to_anthropic(
                request.tool_choice,
                warnings,
            )

        self._apply_generation(raw, request.generation)

        return ConversionResult(value=raw, warnings=warnings)

    def _block_to_anthropic(
        self,
        block: ContentBlock,
        warnings: list[ConversionWarning],
        path: str,
    ) -> dict:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}

        if isinstance(block, ImageBlock):
            if block.source.kind == "base64":
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.source.mime_type,
                        "data": block.source.value,
                    },
                }

            warnings.append(
                ConversionWarning(
                    code="IMAGE_SOURCE_DOWNGRADED",
                    path=path,
                    message="Anthropic image block may require base64 source depending on endpoint.",
                )
            )
            return {
                "type": "text",
                "text": f"[Image: {block.source.value}]",
            }

        if isinstance(block, ToolCallBlock):
            return {
                "type": "tool_use",
                "id": block.call.id,
                "name": block.call.name,
                "input": block.call.arguments,
            }

        if isinstance(block, ToolResultBlock):
            return {
                "type": "tool_result",
                "tool_use_id": block.result.call_id,
                "content": blocks_to_plain_text(block.result.content, warnings, path),
                "is_error": block.result.is_error,
            }

        warnings.append(
            ConversionWarning(
                code="UNSUPPORTED_CONTENT_BLOCK",
                path=path,
                message="Unsupported block for Anthropic.",
            )
        )
        return {"type": "text", "text": "[Unsupported content omitted]"}
```

---

# 18. Gemini Adapter

Gemini 的抽象是：

```txt
contents[]
  role: user / model
  parts[]
    text
    inlineData
    fileData
    functionCall
    functionResponse

systemInstruction
tools.functionDeclarations
generationConfig
```

Gemini 官方 function calling 用于让模型决定调用外部函数，并返回函数参数。([Google AI for Developers][9])

## 18.1 UIR -> Gemini

```python
class GeminiAdapter:
    provider = "google"

    def uir_to_request(self, request: UniversalRequest) -> ConversionResult:
        warnings: list[ConversionWarning] = []

        raw = {
            "contents": [],
        }

        if request.instructions:
            raw["systemInstruction"] = {
                "parts": [
                    {
                        "text": blocks_to_plain_text(
                            request.instructions,
                            warnings,
                            "instructions",
                        )
                    }
                ]
            }

        for idx, msg in enumerate(request.messages):
            content = {
                "role": self._map_role_to_gemini(
                    msg.role,
                    warnings,
                    f"messages[{idx}].role",
                ),
                "parts": [],
            }

            for j, block in enumerate(msg.content):
                content["parts"].append(
                    self._block_to_gemini_part(
                        block,
                        warnings,
                        f"messages[{idx}].content[{j}]",
                    )
                )

            for call in msg.tool_calls:
                content["parts"].append({
                    "functionCall": {
                        "name": call.name,
                        "args": call.arguments,
                    }
                })

            for result in msg.tool_results:
                content["parts"].append({
                    "functionResponse": {
                        "name": result.name or "unknown",
                        "response": {
                            "call_id": result.call_id,
                            "content": blocks_to_plain_text(
                                result.content,
                                warnings,
                                f"messages[{idx}].tool_results",
                            ),
                            "is_error": result.is_error,
                        },
                    }
                })

            raw["contents"].append(content)

        if request.tools:
            raw["tools"] = [{
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema or {
                            "type": "object",
                            "properties": {},
                        },
                    }
                    for tool in request.tools
                    if tool.type == "function"
                ]
            }]

        raw["generationConfig"] = self._generation_to_gemini(request.generation)

        if request.response_format:
            self._apply_response_format_to_gemini(
                raw,
                request.response_format,
                warnings,
            )

        if request.state.previous_response_id or request.state.conversation_id:
            warnings.append(
                ConversionWarning(
                    code="STATE_NOT_SUPPORTED",
                    path="state",
                    message="Gemini standard generateContent request does not use OpenAI-style previous_response_id.",
                )
            )

        return ConversionResult(value=raw, warnings=warnings)

    def _map_role_to_gemini(
        self,
        role: Role,
        warnings: list[ConversionWarning],
        path: str,
    ) -> str:
        if role == "assistant":
            return "model"

        if role in ("user", "tool"):
            return "user"

        if role in ("system", "developer"):
            warnings.append(
                ConversionWarning(
                    code="ROLE_DOWNGRADED",
                    path=path,
                    message="Gemini uses systemInstruction for system/developer instructions.",
                )
            )
            return "user"

        return "user"

    def _block_to_gemini_part(
        self,
        block: ContentBlock,
        warnings: list[ConversionWarning],
        path: str,
    ) -> dict:
        if isinstance(block, TextBlock):
            return {"text": block.text}

        if isinstance(block, ImageBlock):
            if block.source.kind == "base64":
                return {
                    "inlineData": {
                        "mimeType": block.source.mime_type,
                        "data": block.source.value,
                    }
                }

            if block.source.kind == "url":
                return {
                    "fileData": {
                        "mimeType": block.source.mime_type,
                        "fileUri": block.source.value,
                    }
                }

        if isinstance(block, ToolCallBlock):
            return {
                "functionCall": {
                    "name": block.call.name,
                    "args": block.call.arguments,
                }
            }

        if isinstance(block, ToolResultBlock):
            return {
                "functionResponse": {
                    "name": block.result.name or "unknown",
                    "response": {
                        "content": blocks_to_plain_text(
                            block.result.content,
                            warnings,
                            path,
                        )
                    },
                }
            }

        warnings.append(
            ConversionWarning(
                code="UNSUPPORTED_CONTENT_BLOCK",
                path=path,
                message="Unsupported Gemini part.",
            )
        )
        return {"text": "[Unsupported content omitted]"}
```

---

# 19. Amazon Bedrock Adapter

建议优先对接 Bedrock Converse API，因为 AWS 官方说明 Converse API 提供可跨支持消息模型使用的一组统一参数。([AWS 文档][10])

## 19.1 UIR -> Bedrock Converse

```python
class BedrockConverseAdapter:
    provider = "bedrock"

    def uir_to_request(self, request: UniversalRequest) -> ConversionResult:
        warnings: list[ConversionWarning] = []

        raw = {
            "modelId": request.model.name,
            "messages": [],
            "inferenceConfig": {},
        }

        if request.instructions:
            raw["system"] = [
                {
                    "text": blocks_to_plain_text(
                        request.instructions,
                        warnings,
                        "instructions",
                    )
                }
            ]

        for idx, msg in enumerate(request.messages):
            if msg.role in ("system", "developer"):
                raw.setdefault("system", []).append({
                    "text": blocks_to_plain_text(
                        msg.content,
                        warnings,
                        f"messages[{idx}].content",
                    )
                })
                warnings.append(
                    ConversionWarning(
                        code="ROLE_DOWNGRADED",
                        path=f"messages[{idx}].role",
                        message="Bedrock Converse uses top-level system field.",
                    )
                )
                continue

            bedrock_msg = {
                "role": self._map_role(msg.role, warnings, f"messages[{idx}].role"),
                "content": [],
            }

            for j, block in enumerate(msg.content):
                bedrock_msg["content"].append(
                    self._block_to_bedrock(
                        block,
                        warnings,
                        f"messages[{idx}].content[{j}]",
                    )
                )

            for call in msg.tool_calls:
                bedrock_msg["content"].append({
                    "toolUse": {
                        "toolUseId": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                })

            for result in msg.tool_results:
                bedrock_msg["content"].append({
                    "toolResult": {
                        "toolUseId": result.call_id,
                        "content": [
                            {
                                "text": blocks_to_plain_text(
                                    result.content,
                                    warnings,
                                    f"messages[{idx}].tool_results",
                                )
                            }
                        ],
                        "status": "error" if result.is_error else "success",
                    }
                })

            raw["messages"].append(bedrock_msg)

        if request.tools:
            raw["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": {
                                "json": tool.input_schema or {
                                    "type": "object",
                                    "properties": {},
                                }
                            },
                        }
                    }
                    for tool in request.tools
                    if tool.type == "function"
                ]
            }

        self._apply_generation(raw["inferenceConfig"], request.generation)

        return ConversionResult(value=raw, warnings=warnings)

    def _block_to_bedrock(
        self,
        block: ContentBlock,
        warnings: list[ConversionWarning],
        path: str,
    ) -> dict:
        if isinstance(block, TextBlock):
            return {"text": block.text}

        if isinstance(block, ImageBlock):
            if block.source.kind == "bytes":
                return {
                    "image": {
                        "format": self._mime_to_bedrock_image_format(block.source.mime_type),
                        "source": {
                            "bytes": block.source.value,
                        },
                    }
                }

            warnings.append(
                ConversionWarning(
                    code="IMAGE_SOURCE_DOWNGRADED",
                    path=path,
                    message="Bedrock image source often requires bytes.",
                )
            )
            return {"text": f"[Image omitted: {block.source.value}]"}

        if isinstance(block, ToolCallBlock):
            return {
                "toolUse": {
                    "toolUseId": block.call.id,
                    "name": block.call.name,
                    "input": block.call.arguments,
                }
            }

        if isinstance(block, ToolResultBlock):
            return {
                "toolResult": {
                    "toolUseId": block.result.call_id,
                    "content": [
                        {
                            "text": blocks_to_plain_text(
                                block.result.content,
                                warnings,
                                path,
                            )
                        }
                    ],
                    "status": "error" if block.result.is_error else "success",
                }
            }

        warnings.append(
            ConversionWarning(
                code="UNSUPPORTED_CONTENT_BLOCK",
                path=path,
                message="Unsupported Bedrock content block.",
            )
        )
        return {"text": "[Unsupported content omitted]"}
```

---

# 20. Ollama Adapter

Ollama 有两条路：

```txt
1. Ollama native /api/chat
2. Ollama OpenAI-compatible /v1/chat/completions
```

官方说明 Ollama 提供 OpenAI compatibility；同时其 tool calling 文档说明支持 function calling。([Ollama 文档][3])

建议第一版走 OpenAI-compatible，这样可复用 OpenAI Chat Adapter。

```python
class OllamaOpenAICompatAdapter(OpenAIChatAdapter):
    provider = "ollama"

    def uir_to_request(self, request: UniversalRequest) -> ConversionResult:
        result = super().uir_to_request(request)

        if request.state.previous_response_id or request.state.conversation_id:
            result.warnings.append(
                ConversionWarning(
                    code="STATE_NOT_SUPPORTED",
                    path="state",
                    message="Ollama OpenAI-compatible API does not support OpenAI-style stateful Responses fields.",
                )
            )

        return result
```

---

# 21. Cloudflare Workers AI Adapter

Cloudflare Workers AI 支持 OpenAI-compatible `/v1/chat/completions` 和 `/v1/embeddings`。([Cloudflare Docs][7])

第一版同样建议复用 OpenAI Chat Adapter。

```python
class CloudflareWorkersAIAdapter(OpenAIChatAdapter):
    provider = "cloudflare"

    def uir_to_request(self, request: UniversalRequest) -> ConversionResult:
        result = super().uir_to_request(request)

        # Workers AI 的高级能力取决于模型，不要默认承诺全部支持。
        for idx, msg in enumerate(request.messages):
            for j, block in enumerate(msg.content):
                if isinstance(block, (AudioBlock, FileBlock, ThinkingBlock)):
                    result.warnings.append(
                        ConversionWarning(
                            code="UNSUPPORTED_CONTENT_BLOCK",
                            path=f"messages[{idx}].content[{j}]",
                            message="Cloudflare Workers AI OpenAI-compatible endpoint may not support this block type.",
                        )
                    )

        return result
```

---

# 22. OpenAI Assistants Adapter

Assistants API 抽象层较高：

```txt
assistant
thread
message
run
required_action
tool_outputs
```

UIR 不应该模仿 Assistants，而应该把 Assistants 的状态放入：

```python
state.thread_id
state.run_id
state.provider_state
```

## 22.1 Assistants -> UIR

```python
class OpenAIAssistantsAdapter:
    provider = "openai_assistants"

    def request_to_uir(self, raw: dict) -> ConversionResult:
        warnings: list[ConversionWarning] = []

        state = ConversationState(
            thread_id=raw.get("thread_id"),
            run_id=raw.get("run_id"),
            provider_state={
                "assistant_id": raw.get("assistant_id"),
            },
        )

        instructions = []
        if raw.get("instructions"):
            instructions.append(
                TextBlock(type="text", text=raw["instructions"])
            )

        messages = []
        for idx, msg in enumerate(raw.get("messages", [])):
            messages.append(
                Message(
                    id=msg.get("id"),
                    role=self._map_assistant_role(msg.get("role")),
                    content=self._parse_assistant_content(
                        msg.get("content", []),
                        warnings,
                        f"messages[{idx}].content",
                    ),
                    provider={"raw": msg},
                )
            )

        request = UniversalRequest(
            version="uai.v1",
            model=ModelRef(
                provider="openai",
                name=raw.get("model", ""),
            ),
            instructions=instructions,
            messages=messages,
            tools=[self._parse_tool(t) for t in raw.get("tools", [])],
            state=state,
            vendor={"openai_assistants": raw},
        )

        warnings.append(
            ConversionWarning(
                code="API_DEPRECATED_OR_MIGRATING",
                path="provider",
                message="OpenAI Assistants API should be treated as a legacy/source adapter and migrated toward Responses where possible.",
                severity="info",
            )
        )

        return ConversionResult(value=request, warnings=warnings)
```

---

# 23. Stream Event 设计

不同 provider 的流式事件差异极大，建议不要直接互转 provider event，而是转成统一事件。

```python
StreamEventType = Literal[
    "response_started",
    "message_started",
    "content_delta",
    "tool_call_started",
    "tool_call_delta",
    "tool_call_completed",
    "message_completed",
    "response_completed",
    "error",
]


@dataclass
class UniversalStreamEvent:
    type: StreamEventType
    response_id: Optional[str] = None
    message_id: Optional[str] = None
    index: Optional[int] = None

    delta: Optional[ContentBlock] = None
    tool_call: Optional[ToolCall] = None
    usage: Optional["Usage"] = None

    raw: Any = None
```

转换示意：

```python
def stream_provider_to_uir(provider_events: list[dict]) -> list[UniversalStreamEvent]:
    events = []

    for raw in provider_events:
        if is_text_delta(raw):
            events.append(
                UniversalStreamEvent(
                    type="content_delta",
                    delta=TextBlock(type="text", text=extract_text_delta(raw)),
                    raw=raw,
                )
            )

        elif is_tool_call_delta(raw):
            events.append(
                UniversalStreamEvent(
                    type="tool_call_delta",
                    tool_call=extract_tool_call_delta(raw),
                    raw=raw,
                )
            )

        elif is_done(raw):
            events.append(
                UniversalStreamEvent(
                    type="response_completed",
                    usage=extract_usage(raw),
                    raw=raw,
                )
            )

        else:
            events.append(
                UniversalStreamEvent(
                    type="content_delta",
                    delta=ProviderBlock(
                        type="provider_block",
                        provider="unknown",
                        value=raw,
                    ),
                    raw=raw,
                )
            )

    return events
```

---

# 24. Stop Reason 统一

```python
PROVIDER_STOP_REASON_MAP = {
    "openai": {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
    },
    "anthropic": {
        "end_turn": "end_turn",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "tool_use": "tool_calls",
    },
    "gemini": {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    },
    "bedrock": {
        "end_turn": "end_turn",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "tool_use": "tool_calls",
    },
}
```

---

# 25. Usage 设计

```python
@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None

    provider: dict[str, Any] = field(default_factory=dict)
```

不要强行统一所有计费字段。provider 原始 usage 放进：

```python
usage.provider
```

---

# 26. UniversalResponse

```python
@dataclass
class UniversalResponse:
    id: Optional[str]
    model: Optional[str]
    output: list[Message]

    stop_reason: StopReason = "unknown"
    usage: Optional[Usage] = None

    state: ConversationState = field(default_factory=ConversationState)

    raw: Any = None
    vendor: dict[str, Any] = field(default_factory=dict)
```

---

# 27. 示例：工具调用完整链路

## 27.1 UIR Request

```python
request = UniversalRequest(
    version="uai.v1",
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    instructions=[
        TextBlock(type="text", text="你是一个天气助手。")
    ],
    messages=[
        Message(
            role="user",
            content=[
                TextBlock(type="text", text="东京明天天气怎么样？")
            ],
        )
    ],
    tools=[
        ToolDef(
            type="function",
            name="get_weather",
            description="Get weather by city and date.",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["city", "date"],
            },
        )
    ],
    tool_choice="auto",
)
```

## 27.2 模型返回工具调用

```python
assistant_msg = Message(
    role="assistant",
    content=[],
    tool_calls=[
        ToolCall(
            id="call_001",
            name="get_weather",
            arguments={
                "city": "Tokyo",
                "date": "tomorrow",
            },
        )
    ],
)
```

## 27.3 工具结果

```python
tool_msg = Message(
    role="tool",
    content=[],
    tool_results=[
        ToolResult(
            call_id="call_001",
            name="get_weather",
            content=[
                TextBlock(
                    type="text",
                    text='{"temperature": 22, "condition": "rain"}',
                )
            ],
        )
    ],
)
```

## 27.4 第二轮请求

```python
request.messages.extend([assistant_msg, tool_msg])
```

---

# 28. 互转验收标准

## 28.1 L0 验收

以下 provider 任意互转，必须无严重 warning：

```txt
OpenAI Chat
OpenAI Responses
Anthropic
Gemini
Bedrock
Ollama
Cloudflare Workers AI
```

测试输入：

```python
messages = [
    system("You are helpful."),
    user("Hello"),
    assistant("Hi!"),
    user("Explain Python list comprehension."),
]
```

验收：

```txt
1. role 不丢失
2. 文本不丢失
3. generation config 尽量映射
4. stream 可映射
```

## 28.2 L1 工具调用验收

测试：

```python
user -> assistant tool_call -> tool result -> assistant final answer
```

验收：

```txt
1. tool name 不丢
2. tool arguments 不丢
3. tool call id 尽量不丢
4. tool result 能被目标 provider 识别
5. 不支持 tool 的 provider 必须 warning
```

## 28.3 L1 多模态验收

测试：

```python
user content:
- text
- image
```

验收：

```txt
1. 支持 image 的 provider 应正确转换
2. 不支持 image 的 provider 必须 placeholder + warning
3. 不允许静默丢图
```

## 28.4 L2 状态验收

测试：

```python
previous_response_id
conversation_id
thread_id
run_id
```

验收：

```txt
1. OpenAI Responses 应保留 previous_response_id / conversation
2. Assistants 应保留 thread_id / run_id
3. Chat Completions / Ollama / Cloudflare 降级时必须 warning
```

---

# 29. 推荐项目结构

```txt
universal_ai_ir/
  __init__.py

  schema/
    request.py
    response.py
    message.py
    content.py
    tools.py
    warnings.py
    usage.py
    stream.py

  adapters/
    base.py
    openai_chat.py
    openai_responses.py
    openai_assistants.py
    anthropic.py
    gemini.py
    bedrock.py
    ollama.py
    cloudflare.py

  capabilities/
    matrix.py
    validation.py

  conversion/
    converter.py
    lowering.py
    normalization.py

  tests/
    test_l0_text.py
    test_l1_tools.py
    test_l1_multimodal.py
    test_l2_state.py
    fixtures/
      openai_chat.json
      openai_responses.json
      anthropic.json
      gemini.json
      bedrock.json
```

---

# 30. 第一版 MVP 范围

## 必须做

```txt
1. Text messages
2. System / developer / user / assistant / tool roles
3. ContentBlock array
4. Function tools
5. Tool calls
6. Tool results
7. JSON schema response_format
8. Generation config
9. Stop reason
10. Usage
11. Warnings
12. OpenAI Chat adapter
13. OpenAI Responses adapter
14. Anthropic adapter
15. Gemini adapter
16. Bedrock Converse adapter
17. Ollama OpenAI-compatible adapter
18. Cloudflare OpenAI-compatible adapter
```

## 可以后做

```txt
1. Realtime / Live API
2. Audio output
3. Computer use
4. File search
5. Web search
6. MCP tools
7. Prompt cache
8. Safety settings normalization
9. Batch API
10. Fine-grained stream event parity
```

---

# 31. 一句话总结

这个 PRD 的核心不是“做一个更大的 ChatMessage”，而是做一个 **provider-neutral 的语义层**：

```txt
Message 是对话语义
ContentBlock 是多模态语义
ToolCall / ToolResult 是执行语义
State 是会话语义
Vendor 是保真兜底
Warning 是降级透明度
```

这样才能做到从 Anthropic、Gemini、Bedrock、Ollama、OpenAI Responses / Assistants / Chat Completions、Cloudflare Workers AI 之间稳定互转，而不是每新增一个 provider 就重写一套胶水代码。

[1]: https://developers.openai.com/api/docs/assistants/migration?utm_source=chatgpt.com "Assistants migration guide | OpenAI API"
[2]: https://developers.openai.com/api/reference/responses/overview/?utm_source=chatgpt.com "Responses Overview | OpenAI API Reference"
[3]: https://docs.ollama.com/api/openai-compatibility?utm_source=chatgpt.com "OpenAI compatibility"
[4]: https://platform.claude.com/docs/en/api/messages?utm_source=chatgpt.com "Messages - Claude API Reference"
[5]: https://docs.ollama.com/capabilities/tool-calling?utm_source=chatgpt.com "Tool calling"
[6]: https://developers.openai.com/api/docs/guides/structured-outputs?utm_source=chatgpt.com "Structured model outputs | OpenAI API"
[7]: https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/?utm_source=chatgpt.com "OpenAI compatible API endpoints · Cloudflare Workers AI ..."
[8]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview?utm_source=chatgpt.com "Tool use with Claude - Claude API Docs"
[9]: https://ai.google.dev/gemini-api/docs/function-calling?utm_source=chatgpt.com "Function calling with the Gemini API | Google AI for Developers"
[10]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html?utm_source=chatgpt.com "Anthropic Claude Messages API - Amazon Bedrock"
