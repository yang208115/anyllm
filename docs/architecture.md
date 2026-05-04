# 架构设计

## 设计目标

AnyLLM 的核心设计目标是解决 LLM 生态碎片化问题 — 每个 provider 的 API 格式、消息结构、能力集合各不相同，导致应用层需要大量胶水代码适配。

AnyLLM 通过引入 **UIR（Universal Intermediate Representation）** 语义层，将问题从 O(n²) 降低为 O(n)：每新增一个 provider，只需编写一个适配器，而不是对每一对 provider 都编写转换逻辑。

```
不使用 AnyLLM（O(n²) 复杂度）：
  OpenAI ←→ Anthropic
  OpenAI ←→ Gemini
  OpenAI ←→ Bedrock
  Anthropic ←→ Gemini
  Anthropic ←→ Bedrock
  Gemini ←→ Bedrock
  ...

使用 AnyLLM（O(n) 复杂度）：
  OpenAI   ←→ UIR
  Anthropic ←→ UIR
  Gemini    ←→ UIR
  Bedrock   ←→ UIR
  Ollama    ←→ UIR
```

## 三大设计原则

### 原则一：万物皆 Block

所有消息内容统一表示为 `List[ContentBlock]`，不使用 `content: str` 的简化格式。这确保了多模态内容（文本、图片、音频、工具调用、工具结果等）可以在所有 provider 之间无损转换。

```
Message.content: List[ContentBlock]
                      │
    ┌─────────────────┼──────────────────┐
    │                 │                  │
  TextBlock      ImageBlock       ToolCallBlock
  "Hello!"       (base64)        get_weather(...)
```

### 原则二：极简角色

UIR 保留 5 种角色（`system` / `developer` / `user` / `assistant` / `tool`），但通过拦截器和适配器自动处理角色映射差异：

| UIR 角色 | OpenAI Chat | Anthropic | Gemini | Bedrock |
|----------|-------------|-----------|--------|---------|
| system | system message | system 字段（顶层） | systemInstruction | system 字段 |
| developer | system/developer | 合并到 system | 合并到 systemInstruction | 合并到 system |
| user | user | user | user | user |
| assistant | assistant | assistant | model | assistant |
| tool | tool | user (tool_result block) | user (functionResponse) | user (toolResult) |

### 原则三：降级透明

所有转换函数返回 `ConversionResult[T]` 而非裸返回 `dict`。当发生能力降级（如 Anthropic 不支持 URL 图片、OpenAI 不支持 tool_result blocks）时，会生成 `ConversionWarning`，调用方可以决定如何处理。

## 核心组件关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AnyLLMGateway                               │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────────────────┐               │
│  │ ProviderConfig│   │     UniversalConverter        │               │
│  │  ├─ Adapter   │   │  ┌──────────┐ ┌───────────┐  │               │
│  │  ├─ API Key   │──▶│  │ Adapters │ │Interceptors│  │               │
│  │  ├─ Base URL  │   │  │  Dict    │ │   List     │  │               │
│  │  └─ Timeout   │   │  └──────────┘ └───────────┘  │               │
│  └──────────────┘   └──────────────────────────────┘               │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │   Router     │  路由策略（默认按 model.provider / 模型名推断）      │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**层次关系**：

| 组件 | 职责 | 依赖 |
|------|------|------|
| **Schema** (`schema/`) | 定义 UIR 数据模型 | 仅 Pydantic |
| **BaseAdapter** (`adapters/base.py`) | 适配器抽象接口 | Schema |
| **Adapters** (`adapters/*.py`) | 厂商适配器实现 | BaseAdapter, Schema, Lowering |
| **Lowering** (`conversion/lowering.py`) | 降级工具函数 | Schema |
| **Interceptors** (`interceptors.py`) | 中间件实现 | BaseInterceptor, Schema |
| **Converter** (`conversion/converter.py`) | 适配器 + 拦截器编排 | BaseAdapter, BaseInterceptor |
| **Gateway** (`gateway.py`) | 网关入口，HTTP 调用 | Converter, httpx |

## 数据流

### 完整请求流（Gateway 模式）

```
                    AnyLLMGateway.chat_completions()
                           │
                    ┌──────┴──────┐
                    │   Router    │  ① 确定目标 provider
                    └──────┬──────┘
                           │
         ┌─────────────────┴─────────────────┐
         │     Interceptor Pipeline           │  ② UIR 层面预处理
         │  ┌───────────────────────────┐     │
         │  │ ImageResolutionInterceptor│     │  - URL 图片 → base64
         │  │ RoleConsolidationIntrcptr │     │  - 角色整理/合并
         │  │ CustomInterceptor_1       │     │  - 用户自定义逻辑
         │  │ CustomInterceptor_2       │     │
         │  └───────────────────────────┘     │
         └─────────────────┬─────────────────┘
                           │
              UIR UniversalRequest (处理后)
                           │
                    ┌──────┴──────┐
                    │  Adapter    │  ③ UIR → provider dict
                    │ uir_to_req │
                    └──────┬──────┘
                           │
                    provider 请求 dict
                           │
                    ┌──────┴──────┐
                    │  HTTP POST  │  ④ 调用 provider API
                    └──────┬──────┘
                           │
                    provider 响应 dict
                           │
                    ┌──────┴──────┐
                    │  Adapter    │  ⑤ provider dict → UIR
                    │ resp_to_uir│
                    └──────┬──────┘
                           │
                  GatewayResult
                  ├── response: UniversalResponse
                  └── warnings: List[ConversionWarning]
```

### 纯转换流（Converter 模式）

```
  Provider A 请求 dict
         │
  ┌──────┴──────┐
  │  Adapter A  │  request_to_uir()
  └──────┬──────┘
         │
  UIR UniversalRequest
         │
  ┌──────┴──────┐
  │ Interceptors│  run_interceptors()
  └──────┬──────┘
         │
  UIR UniversalRequest (处理后)
         │
  ┌──────┴──────┐
  │  Adapter B  │  uir_to_request()
  └──────┬──────┘
         │
  Provider B 请求 dict
```

## 模块依赖图

```
schema/
├── content.py      ←─ 无外部依赖（只用 Pydantic）
├── message.py      ←─ content.py
├── tools.py        ←─ 无外部依赖
├── usage.py        ←─ 无外部依赖
├── warnings.py     ←─ 无外部依赖
├── request.py      ←─ content, message, tools
├── response.py     ←─ message, request, usage
└── stream.py       ←─ content, usage

adapters/
├── base.py         ←─ schema.request, schema.response, schema.warnings
├── openai_chat.py  ←─ base.py, schema.*, conversion.lowering
└── anthropic.py    ←─ base.py, schema.*, conversion.lowering

conversion/
├── lowering.py     ←─ schema.content, schema.warnings
└── converter.py    ←─ adapters.base, schema.*

interceptors.py     ←─ adapters.base, schema.*

gateway.py          ←─ conversion.converter, adapters.base, schema.*
```

## 多态反序列化

AnyLLM 大量使用 Pydantic v2 的 **Discriminated Union** 实现多态反序列化：

```python
ContentBlock = Annotated[
    Union[TextBlock, ImageBlock, AudioBlock, ...],
    Field(discriminator="type"),
]
```

根据 JSON 数据中的 `type` 字段自动路由到对应的子类：

```python
data = {"type": "text", "text": "Hello"}
# Pydantic 自动反序列化为 TextBlock

data = {"type": "image", "source": {"kind": "url", "value": "..."}}
# Pydantic 自动反序列化为 ImageBlock
```

同样的模式用于 `ToolChoice` 和 `ResponseFormat`：

```python
ToolChoice = Annotated[
    Union[AutoToolChoice, NoneToolChoice, RequiredToolChoice, SpecificToolChoice],
    Field(discriminator="type"),
]

ResponseFormat = Annotated[
    Union[TextResponseFormat, JsonObjectResponseFormat, JsonSchemaResponseFormat],
    Field(discriminator="type"),
]
```

## 循环依赖处理

`ToolResult.content` 的类型是 `List[ContentBlock]`，而 `ToolResultBlock.result` 的类型是 `ToolResult`，形成循环引用。解决方案：

1. 使用 `from __future__ import annotations` 使所有类型注解成为字符串（延迟求值）
2. 将 `ToolCall` 和 `ToolResult` 定义在 `content.py` 中（与 `ContentBlock` 同模块）
3. 在模块末尾调用 `model_rebuild()` 让 Pydantic 解析前向引用：

```python
# content.py 末尾
ToolResult.model_rebuild()
ToolCallBlock.model_rebuild()
ToolResultBlock.model_rebuild()
```

## 无状态设计

AnyLLM 的所有核心组件（Adapter、Interceptor、Converter、Gateway）都是无状态的：

- **Adapter** 实例不持有任何请求相关状态，同一个实例可在多个请求间复用
- **Interceptor** 不保存跨请求状态（除非用户自定义的拦截器需要）
- **Converter** 只维护适配器和拦截器的注册表，不保存请求状态
- **Gateway** 只维护 provider 配置，不保存会话状态

这使得所有组件可以安全地在多个 `asyncio` task 之间共享。

## 扩展点

| 扩展点 | 方式 | 参考文档 |
|--------|------|---------|
| 新增 provider | 继承 `BaseAdapter`，实现 4 个转换方法 | [适配器开发指南](adapters.md) |
| 自定义中间件 | 继承 `BaseInterceptor` 或使用 `@interceptor` 装饰器 | [拦截器开发指南](interceptors.md) |
| 自定义路由 | 调用 `gateway.set_router(fn)` | [网关使用指南](gateway.md) |
| 厂商特定字段 | 使用 `vendor` / `provider` 透传区 | [UIR 模型参考](uir-reference.md) |
