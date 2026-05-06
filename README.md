# AnyLLM

**通用大模型网关核心库** — 一个 provider-neutral 的语义层，支持 OpenAI / Anthropic / Gemini / Bedrock / Ollama 等 LLM API 之间的无损互转。

```
Provider A Request ──► UIR (Universal Intermediate Representation) ──► Provider B Request
Provider A Response ◄── UIR ◄── Provider B Response
```

## 解决什么问题

主流 LLM API 的抽象层级各不相同：

| API | 抽象风格 |
|-----|---------|
| OpenAI Chat Completions | 简单 chat message，tool_calls 在消息顶层 |
| OpenAI Responses | 富格式，stateful，内置工具 |
| Anthropic Messages | content block 模型，tool_use/tool_result 嵌入 content |
| Google Gemini | parts 模型，functionCall/functionResponse |
| Amazon Bedrock Converse | 统一 content + toolUse/toolResult |
| Ollama / Cloudflare | OpenAI-compatible 子集 |

AnyLLM 设计了一个 **Universal Intermediate Representation (UIR)**，使得不同 provider 之间可以稳定互转，而不是每新增一个 provider 就重写一套胶水代码。

## 安装

```bash
# 基础安装（仅格式转换，不调用 API）
pip install anyllm

# 包含 HTTP 客户端（调用 provider API + 图片自动下载）
pip install anyllm[http]

# 开发环境
pip install anyllm[dev]
```

从源码安装：

```bash
git clone https://github.com/anyllm/anyllm.git
cd anyllm
pip install -e ".[dev]"
```

## 快速上手

### 1. 格式转换（不调用 API）

```python
from anyllm import (
    UniversalConverter,
    OpenAIChatAdapter,
    AnthropicAdapter,
    ImageResolutionInterceptor,
    RoleConsolidationInterceptor,
)

# 初始化转换器
converter = UniversalConverter()
converter.register_adapter("openai_chat", OpenAIChatAdapter())
converter.register_adapter("anthropic", AnthropicAdapter())
converter.register_interceptor(ImageResolutionInterceptor())
converter.register_interceptor(RoleConsolidationInterceptor())

# OpenAI 请求 → Anthropic 请求
openai_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
}

import asyncio

result = asyncio.run(
    converter.convert_request("openai_chat", "anthropic", openai_request)
)

print(result.value)
# {
#     "model": "gpt-4o",
#     "max_tokens": 1024,
#     "system": "You are a helpful assistant.",
#     "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}],
#     "temperature": 0.7,
# }

# 查看转换警告
for w in result.warnings:
    print(f"[{w.severity}] {w.code}: {w.message}")
```

### 2. 使用 UIR 统一格式构建请求

```python
from anyllm import (
    UniversalRequest, ModelRef, Message, TextBlock, ImageBlock,
    MediaSource, ToolDef, ToolCall, ToolResult,
    GenerationConfig, AutoToolChoice,
)

request = UniversalRequest(
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    instructions=[TextBlock(text="你是一个天气助手。")],
    messages=[
        Message.user_text("东京明天天气怎么样？"),
        Message(
            role="assistant",
            content=[TextBlock(text="让我查一下。")],
            tool_calls=[
                ToolCall(id="call_1", name="get_weather",
                         arguments={"city": "Tokyo", "date": "tomorrow"})
            ],
        ),
        Message.tool_result("call_1", '{"temperature": 22, "condition": "rain"}'),
    ],
    tools=[
        ToolDef(
            name="get_weather",
            description="获取指定城市的天气信息",
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
    generation=GenerationConfig(temperature=0.7, max_output_tokens=1024),
)
```

### 3. 网关模式（调用 API）

```python
from anyllm import (
    AnyLLMGateway, ProviderConfig,
    OpenAIChatAdapter, AnthropicAdapter, GeminiAdapter,
    ImageResolutionInterceptor, RoleConsolidationInterceptor,
    UniversalRequest, ModelRef, Message,
)

gateway = AnyLLMGateway()

# 注册 provider
gateway.register_provider("openai_chat", ProviderConfig(
    adapter=OpenAIChatAdapter(),
    api_base="https://api.openai.com/v1",
    api_key="sk-xxx",
))
gateway.register_provider("anthropic", ProviderConfig(
    adapter=AnthropicAdapter(),
    api_base="https://api.anthropic.com",
    api_key="sk-ant-xxx",
))
gateway.register_provider("google", ProviderConfig(
    adapter=GeminiAdapter(),
    api_base="https://generativelanguage.googleapis.com",
    api_key="$GOOGLE_API_KEY",
))

# 注册拦截器
gateway.register_interceptor(ImageResolutionInterceptor())
gateway.register_interceptor(RoleConsolidationInterceptor())

# 构建请求并调用
request = UniversalRequest(
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    messages=[Message.user_text("Hello!")],
)

import asyncio

async def main():
    result = await gateway.chat_completions(request)
    print(result.response.output[0].content[0].text)
    print(f"Tokens: {result.response.usage}")

asyncio.run(main())
```

### 4. 自定义拦截器

```python
from anyllm import interceptor, TextBlock, FunctionInterceptor

# 方式一：装饰器（最简洁）
@interceptor("add_watermark", only_for={"openai_chat", "anthropic"})
async def add_watermark(request, target_provider):
    request.instructions.append(TextBlock(text="Powered by AnyLLM"))
    return request

gateway.register_interceptor(add_watermark)

# 方式二：函数式
async def log_request(request, target_provider):
    print(f"Sending to {target_provider}: {len(request.messages)} messages")
    return request

gateway.register_interceptor(
    FunctionInterceptor("logger", log_request)
)

# 方式三：完整类（需要状态管理时使用）
from anyllm import BaseInterceptor

class RateLimiter(BaseInterceptor):
    def __init__(self, max_rpm: int = 60):
        self._max_rpm = max_rpm

    @property
    def name(self) -> str:
        return "rate_limiter"

    async def process(self, request, target_provider):
        # 限流逻辑...
        return request
```

## 示例脚本

- `example/01_convert_openai_to_anthropic.py`：OpenAI 请求转换为 Anthropic 请求
- `example/02_step_by_step_conversion.py`：分步执行 request/response 转换
- `example/03_uir_with_tools.py`：UIR 下的 tool call / tool result
- `example/04_custom_interceptor.py`：自定义拦截器
- `example/05_gateway_convert_only.py`：Gateway 仅转换模式
- `example/06_gateway_call_api.py`：Gateway 非流式 API 调用
- `example/07_gateway_stream_events.py`：Gateway 流式事件消费（OpenAI / Anthropic）

## 核心架构

### UIR 数据模型

```
UniversalRequest
├── model: ModelRef (provider + name)
├── instructions: List[ContentBlock]        # System Prompt，抽离到顶层
├── messages: List[Message]                 # 对话历史
│   └── Message
│       ├── role: user | assistant | tool | system | developer
│       ├── content: List[ContentBlock]     # 万物皆 Block
│       ├── tool_calls: List[ToolCall]      # 顶层工具调用（OpenAI 风格）
│       └── tool_results: List[ToolResult]  # 顶层工具结果（OpenAI 风格）
├── tools: List[ToolDef]
├── tool_choice: ToolChoice
├── response_format: ResponseFormat
├── generation: GenerationConfig
├── state: ConversationState                # L2 有状态会话
└── vendor: Dict[str, Any]                  # 厂商特定字段透传
```

### ContentBlock 类型（9 种）

| 类型 | 说明 | 适用场景 |
|-----|------|---------|
| `TextBlock` | 纯文本 | 所有 provider |
| `ImageBlock` | 图片（url/base64/file_id/bytes） | Vision 模型 |
| `AudioBlock` | 音频 | Gemini, OpenAI Realtime |
| `FileBlock` | 文件引用 | OpenAI Responses, Gemini |
| `ThinkingBlock` | 推理过程 | Anthropic extended thinking |
| `RefusalBlock` | 模型拒绝 | OpenAI safety |
| `ToolCallBlock` | 工具调用（content block 形式） | Anthropic/Gemini/Bedrock |
| `ToolResultBlock` | 工具结果（content block 形式） | Anthropic/Gemini/Bedrock |
| `ProviderBlock` | 厂商私有块透传 | 未知类型兜底 |

### 内置拦截器

| 拦截器 | 功能 |
|-------|------|
| `ImageResolutionInterceptor` | 自动下载 URL 图片并转为 base64（Anthropic/Bedrock 需要） |
| `RoleConsolidationInterceptor` | 合并连续同角色消息、提升 system 到 instructions、tool→user 合并 |

### 能力分层

| 层级 | 能力 | 覆盖范围 |
|-----|------|---------|
| **L0** | 纯文本聊天、system prompt、stream | 所有 provider |
| **L1** | 多模态、工具调用、结构化输出、usage | 大部分 provider |
| **L2** | 有状态会话、内置工具、developer 角色 | OpenAI Responses 等 |

### 转换警告（降级透明度）

所有转换方法返回 `ConversionResult[T]`，包含 `warnings` 列表：

```python
result = converter.convert_request("openai_chat", "anthropic", request)

for w in result.warnings:
    print(f"[{w.severity}] {w.code} at {w.path}: {w.message}")
    # [warning] STATE_NOT_SUPPORTED at state: Anthropic 不支持有状态会话
    # [info] VENDOR_FIELD_PASSTHROUGH at vendor.openai: ...
```

典型 warning code：

| Code | 含义 |
|------|------|
| `UNSUPPORTED_MODALITY` | 目标 provider 不支持该模态（图片/音频） |
| `STATE_NOT_SUPPORTED` | 目标 provider 不支持有状态会话 |
| `ROLE_DOWNGRADED` | 角色被合并或降级 |
| `IMAGE_SOURCE_DOWNGRADED` | 图片来源格式不兼容 |
| `RESPONSE_FORMAT_DOWNGRADED` | 结构化输出格式降级 |
| `TOOL_CHOICE_DOWNGRADED` | 工具选择策略降级 |
| `INVALID_TOOL_ARGUMENTS_JSON` | 工具参数 JSON 解析失败 |

## 项目结构

```
anyllm/
├── __init__.py                 # 顶层导出
├── gateway.py                  # AnyLLMGateway 网关入口
├── interceptors.py             # 内置拦截器 + FunctionInterceptor
├── schema/                     # UIR 数据模型（Pydantic v2）
│   ├── content.py              # ContentBlock 体系（9 种）
│   ├── message.py              # Message + Role
│   ├── request.py              # UniversalRequest
│   ├── response.py             # UniversalResponse
│   ├── stream.py               # UniversalStreamEvent
│   ├── tools.py                # ToolDef + ToolChoice + ResponseFormat
│   ├── usage.py                # Usage
│   └── warnings.py             # ConversionWarning + ConversionResult
├── adapters/                   # 厂商适配器
│   ├── base.py                 # BaseAdapter + BaseInterceptor + ProviderCapabilities
│   ├── openai_chat.py          # OpenAI Chat Completions
│   ├── anthropic.py            # Anthropic Messages API
│   └── gemini.py               # Google Gemini generateContent API
├── conversion/                 # 转换层
│   ├── converter.py            # UniversalConverter 编排器
│   └── lowering.py             # 降级工具函数
└── capabilities/               # 能力矩阵
    └── matrix.py               # 7 个 provider 的预设能力声明
```

## 技术栈

- **Python** 3.10+
- **Pydantic v2** — 数据校验与多态反序列化（Discriminated Union）
- **httpx** — 异步 HTTP 客户端（可选，用于 API 调用和图片下载）
- **asyncio** — 原生异步支持

## 文档

完整的开发文档位于 [`docs/`](docs/) 目录：

| 文档 | 内容 |
|------|------|
| [文档索引](docs/index.md) | 文档导航与核心概念速查 |
| [快速上手](docs/quickstart.md) | 安装、配置、4 个使用场景 |
| [架构设计](docs/architecture.md) | 整体架构、核心组件、数据流、设计原则 |
| [UIR 模型参考](docs/uir-reference.md) | 所有 Pydantic 模型的完整字段参考 |
| [适配器开发指南](docs/adapters.md) | 如何编写新的 provider 适配器 |
| [拦截器开发指南](docs/interceptors.md) | 自定义拦截器的三种方式 |
| [网关使用指南](docs/gateway.md) | AnyLLMGateway 完整 API |
| [转换警告参考](docs/warnings.md) | 所有 warning code 与处理策略 |
| [能力矩阵](docs/capabilities.md) | L0/L1/L2 能力分层与 provider 对照表 |

## 路线图

- [x] UIR 核心数据模型（9 种 ContentBlock）
- [x] OpenAI Chat Completions 适配器（双向）
- [x] Anthropic Messages API 适配器（双向）
- [x] ImageResolutionInterceptor
- [x] RoleConsolidationInterceptor
- [x] 自定义拦截器（FunctionInterceptor + @decorator）
- [x] AnyLLMGateway 网关入口
- [x] Google Gemini 适配器
- [x] 流式响应（UniversalStreamEvent，已支持 OpenAI/Anthropic 网关流式事件）
- [ ] 完整测试套件（L0/L1/L2 验收）

## License

GPL V3.0
