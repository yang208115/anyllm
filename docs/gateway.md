# 网关使用指南

本文档介绍 `AnyLLMGateway` 的完整 API 和使用方式。

## 概述

`AnyLLMGateway` 是 AnyLLM 的最高层入口，在 `UniversalConverter` 之上增加了：

- **Provider 路由** — 根据模型或自定义规则选择 provider
- **HTTP 客户端管理** — 自动调用 provider API
- **API Key / Base URL 管理** — 集中管理多个 provider 的配置

| 组件 | 职责 | 需要网络 |
|------|------|---------|
| `UniversalConverter` | 纯格式转换 | 否 |
| `AnyLLMGateway` | 格式转换 + API 调用 | 是 |

如果只需要做格式转换（不调用 API），直接使用 `UniversalConverter` 即可。

## 初始化

```python
from anyllm import AnyLLMGateway

gateway = AnyLLMGateway()
```

## Provider 注册

### ProviderConfig

每个 provider 通过 `ProviderConfig` 配置适配器和 API 参数：

```python
from anyllm import ProviderConfig, OpenAIChatAdapter, AnthropicAdapter

config = ProviderConfig(
    adapter=OpenAIChatAdapter(),      # 厂商适配器实例
    api_base="https://api.openai.com/v1",  # API 基础 URL
    api_key="sk-xxx",                 # API Key
    default_headers={"X-Custom": "value"},  # 额外请求头
    timeout=60.0,                     # HTTP 超时（秒）
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `adapter` | `BaseAdapter` | **必填** | 厂商适配器实例 |
| `api_base` | `str \| None` | `None` | API 基础 URL |
| `api_key` | `str \| None` | `None` | API Key |
| `default_headers` | `Dict[str, str]` | `{}` | 额外请求头 |
| `timeout` | `float` | `60.0` | HTTP 超时（秒） |

### 注册多个 Provider

```python
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

# 也可以注册 OpenAI 兼容的自定义端点
gateway.register_provider("my_endpoint", ProviderConfig(
    adapter=OpenAIChatAdapter(),
    api_base="https://my-llm-proxy.com/v1",
    api_key="my-key",
))
```

### 查看已注册 Provider

```python
print(gateway.registered_providers)
# ['openai_chat', 'anthropic', 'my_endpoint']
```

## 路由策略

Gateway 需要根据请求确定目标 provider。路由按以下优先级执行：

### 优先级 1：自定义路由函数

```python
def my_router(request):
    """根据模型名称路由。"""
    name = request.model.name.lower()
    if "claude" in name:
        return "anthropic"
    if "gpt" in name or "o1" in name:
        return "openai_chat"
    return "my_endpoint"  # 兜底

gateway.set_router(my_router)
```

### 优先级 2：model.provider 字段

如果请求中明确指定了 `model.provider`，Gateway 会直接使用：

```python
request = UniversalRequest(
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    messages=[...],
)
# 自动路由到 "anthropic"
```

支持模糊匹配：`"openai"` 可以匹配 `"openai_chat"`。

### 优先级 3：模型名称推断

如果 `model.provider` 为 `"unknown"`，Gateway 会从模型名称推断：

| 名称关键词 | 路由到 |
|-----------|--------|
| `gpt` / `o1` / `o3` | `openai_chat` |
| `claude` | `anthropic` |
| `gemini` | `google` |
| `llama` / `mistral` / `qwen` | `ollama` |

### 优先级 4：兜底

如果只注册了一个 provider，直接使用该 provider。否则抛出 `ValueError`。

### 强制指定 Provider

在 `chat_completions` 和 `convert_only` 中可以强制指定 provider，跳过路由：

```python
result = await gateway.chat_completions(
    request,
    provider="openai_chat",  # 强制使用 openai_chat
)
```

## 核心方法

### chat_completions

执行完整的请求链路：路由 → 拦截器 → 转换 → HTTP 调用 → 响应解析。

```python
async def chat_completions(
    self,
    request: UniversalRequest,
    *,
    provider: Optional[str] = None,
    http_client: Optional[Any] = None,
) -> GatewayResult
```

| 参数 | 说明 |
|------|------|
| `request` | UIR 统一请求对象 |
| `provider` | 可选，强制指定 provider |
| `http_client` | 可选，外部 httpx.AsyncClient |

返回 `GatewayResult`。

**使用示例**：

```python
import asyncio
from anyllm import (
    AnyLLMGateway, ProviderConfig,
    OpenAIChatAdapter, UniversalRequest,
    ModelRef, Message,
)

gateway = AnyLLMGateway()
gateway.register_provider("openai_chat", ProviderConfig(
    adapter=OpenAIChatAdapter(),
    api_base="https://api.openai.com/v1",
    api_key="sk-xxx",
))

request = UniversalRequest(
    model=ModelRef(provider="openai", name="gpt-4o"),
    messages=[Message.user_text("Hello!")],
)

async def main():
    result = await gateway.chat_completions(request)
    print(result.response.output[0].content[0].text)

asyncio.run(main())
```

### convert_only

只做格式转换（UIR → provider dict），不调用 API。

```python
async def convert_only(
    self,
    request: UniversalRequest,
    *,
    target_provider: Optional[str] = None,
) -> ConversionResult[Dict[str, Any]]
```

**使用示例**：

```python
result = await gateway.convert_only(
    request,
    target_provider="anthropic",
)

print(result.value)    # Anthropic 格式的请求 dict
print(result.warnings) # 转换警告列表
```

**适用场景**：
- 预览转换结果（调试）
- 与自己的 HTTP 客户端集成
- 测试拦截器和适配器

## GatewayResult

`chat_completions` 的返回类型。

| 属性 | 类型 | 说明 |
|------|------|------|
| `response` | `UniversalResponse` | 统一响应对象 |
| `warnings` | `List[ConversionWarning]` | 整个链路的所有警告 |
| `provider` | `str` | 实际使用的 provider |
| `raw_request` | `Dict \| None` | 发给 provider 的原始请求（调试用） |
| `raw_response` | `Dict \| None` | provider 返回的原始响应（调试用） |
| `has_errors` | `bool` | 是否含 error 级别警告 |

```python
result = await gateway.chat_completions(request)

# 读取响应
for msg in result.response.output:
    for block in msg.content:
        if hasattr(block, 'text'):
            print(block.text)

# 检查 token 用量
if result.response.usage:
    print(f"输入: {result.response.usage.input_tokens}")
    print(f"输出: {result.response.usage.output_tokens}")

# 检查停止原因
print(f"停止原因: {result.response.stop_reason}")

# 检查警告
if result.has_errors:
    print("转换过程中有错误！")
for w in result.warnings:
    print(f"[{w.severity}] {w.code} at {w.path}: {w.message}")

# 调试：查看原始请求/响应
print(result.raw_request)
print(result.raw_response)
```

## HTTP 请求构造

Gateway 根据 provider 类型自动构造不同的请求 URL 和 Headers：

### OpenAI / OpenAI 兼容

```
POST {api_base}/chat/completions
Headers:
  Content-Type: application/json
  Authorization: Bearer {api_key}
```

如果 `api_base` 已包含 `/v1`（如 `https://api.openai.com/v1`），不会重复添加。

### Anthropic

```
POST {api_base}/v1/messages
Headers:
  Content-Type: application/json
  x-api-key: {api_key}
  anthropic-version: 2023-06-01
```

### Google Gemini

```
POST {api_base}/v1beta/models/{model}:generateContent?key={api_key}
Headers:
  Content-Type: application/json
```

### 自定义 HTTP 客户端

可以传入自己的 `httpx.AsyncClient`，适用于需要代理、自定义 TLS 等场景：

```python
import httpx

async with httpx.AsyncClient(
    proxies="http://proxy:8080",
    verify="/path/to/cert.pem",
) as client:
    result = await gateway.chat_completions(
        request,
        http_client=client,
    )
```

## 拦截器注册

Gateway 内部维护一个 `UniversalConverter`，拦截器注册委托给它：

```python
from anyllm import (
    ImageResolutionInterceptor,
    RoleConsolidationInterceptor,
    interceptor,
    TextBlock,
)

# 注册内置拦截器
gateway.register_interceptor(ImageResolutionInterceptor())
gateway.register_interceptor(RoleConsolidationInterceptor())

# 注册自定义拦截器
@interceptor("my_middleware")
async def my_middleware(request, target_provider):
    request.instructions.append(TextBlock(text="Custom instruction"))
    return request

gateway.register_interceptor(my_middleware)

# 插入到指定位置
gateway.register_interceptor(my_middleware, position=0)

# 查看已注册拦截器
print(gateway.registered_interceptors)
# ['my_middleware', 'image_resolution', 'role_consolidation']
```

## 完整示例：多 Provider 网关

```python
import asyncio
from anyllm import (
    AnyLLMGateway,
    ProviderConfig,
    OpenAIChatAdapter,
    AnthropicAdapter,
    ImageResolutionInterceptor,
    RoleConsolidationInterceptor,
    UniversalRequest,
    ModelRef,
    Message,
    TextBlock,
    ImageBlock,
    MediaSource,
    ToolDef,
    ToolCall,
    GenerationConfig,
    interceptor,
)

# ---- 初始化 ----
gateway = AnyLLMGateway()

# ---- 注册 Provider ----
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

# ---- 注册拦截器 ----
gateway.register_interceptor(ImageResolutionInterceptor(timeout=15.0))
gateway.register_interceptor(RoleConsolidationInterceptor())

@interceptor("request_logger")
async def request_logger(request, target_provider):
    print(f"[LOG] {request.model.name} -> {target_provider}")
    return request

gateway.register_interceptor(request_logger)

# ---- 自定义路由 ----
def smart_router(request):
    name = request.model.name.lower()
    if "claude" in name:
        return "anthropic"
    return "openai_chat"

gateway.set_router(smart_router)

# ---- 调用 ----
async def main():
    # 请求 1：调用 OpenAI
    r1 = UniversalRequest(
        model=ModelRef(name="gpt-4o"),
        messages=[Message.user_text("什么是量子计算？")],
        generation=GenerationConfig(temperature=0.7, max_output_tokens=500),
    )
    result1 = await gateway.chat_completions(r1)
    print(f"[OpenAI] {result1.response.output[0].content[0].text[:100]}...")

    # 请求 2：调用 Anthropic
    r2 = UniversalRequest(
        model=ModelRef(name="claude-sonnet-4-5"),
        messages=[
            Message(role="user", content=[
                TextBlock(text="描述这张图片"),
                ImageBlock(source=MediaSource(kind="url", value="https://example.com/img.jpg")),
            ]),
        ],
        generation=GenerationConfig(max_output_tokens=1024),
    )
    result2 = await gateway.chat_completions(r2)
    print(f"[Anthropic] {result2.response.output[0].content[0].text[:100]}...")

    # 请求 3：仅格式转换（不调用 API）
    r3 = UniversalRequest(
        model=ModelRef(name="gpt-4o"),
        messages=[Message.user_text("Test")],
        tools=[ToolDef(name="calculator", description="计算器")],
    )
    preview = await gateway.convert_only(r3, target_provider="anthropic")
    print(f"[Preview] Anthropic format: {preview.value}")
    for w in preview.warnings:
        print(f"  [{w.severity}] {w.code}: {w.message}")

asyncio.run(main())
```
