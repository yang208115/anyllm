# 快速上手

本文介绍如何安装 AnyLLM 并完成第一个请求。

## 安装

```bash
# 基础安装（仅格式转换，不调用 API）
pip install anyllm

# 包含 HTTP 客户端（调用 provider API + 图片自动下载）
pip install anyllm[http]

# 开发环境（含 pytest、ruff 等）
pip install anyllm[dev]
```

从源码安装：

```bash
git clone https://github.com/anyllm/anyllm.git
cd anyllm
pip install -e ".[dev]"
```

## 场景一：纯格式转换

最基础的用法 — 将一个 provider 的请求格式转换为另一个 provider 的格式，不调用任何 API。

```python
import asyncio
from anyllm import (
    UniversalConverter,
    OpenAIChatAdapter,
    AnthropicAdapter,
    ImageResolutionInterceptor,
    RoleConsolidationInterceptor,
)

# 1. 初始化转换器
converter = UniversalConverter()
converter.register_adapter("openai_chat", OpenAIChatAdapter())
converter.register_adapter("anthropic", AnthropicAdapter())

# 2. 注册拦截器（可选，按执行顺序注册）
converter.register_interceptor(ImageResolutionInterceptor())
converter.register_interceptor(RoleConsolidationInterceptor())

# 3. 准备 OpenAI 格式的请求
openai_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
}

# 4. 一步到位：OpenAI → Anthropic
result = asyncio.run(
    converter.convert_request("openai_chat", "anthropic", openai_request)
)

print(result.value)
# 输出 Anthropic 格式的请求 dict：
# {
#     "model": "gpt-4o",
#     "max_tokens": 1024,
#     "system": "You are a helpful assistant.",
#     "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}],
#     "temperature": 0.7,
# }

# 5. 查看转换警告
for w in result.warnings:
    print(f"[{w.severity}] {w.code}: {w.message}")
```

### 分步转换

如果需要在中间插入自定义逻辑，可以分步执行：

```python
# 步骤 1：OpenAI dict → UIR
uir_result = converter.request_to_uir("openai_chat", openai_request)
uir_request = uir_result.value

# 步骤 2：在 UIR 层面自定义修改
uir_request.generation.temperature = 0.5

# 步骤 3：执行拦截器管道
processed = asyncio.run(
    converter.run_interceptors(uir_request, "anthropic")
)

# 步骤 4：UIR → Anthropic dict
target_result = converter.uir_to_request("anthropic", processed)
```

## 场景二：使用 UIR 构建请求

直接用 UIR 模型构建请求，无需先写某个 provider 的原始格式：

```python
from anyllm import (
    UniversalRequest,
    ModelRef,
    Message,
    TextBlock,
    ImageBlock,
    MediaSource,
    ToolDef,
    ToolCall,
    GenerationConfig,
)

# 构建多模态 + 工具调用的请求
request = UniversalRequest(
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    instructions=[TextBlock(text="你是一个天气助手。")],
    messages=[
        # 用户消息（带图片）
        Message(
            role="user",
            content=[
                TextBlock(text="这张图片是什么地方？那里天气如何？"),
                ImageBlock(
                    source=MediaSource(
                        kind="url",
                        value="https://example.com/tokyo.jpg",
                    )
                ),
            ],
        ),
        # 助手消息 + 工具调用
        Message(
            role="assistant",
            content=[TextBlock(text="这是东京塔。让我查一下天气。")],
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_weather",
                    arguments={"city": "Tokyo"},
                )
            ],
        ),
        # 工具结果（快捷构造）
        Message.tool_result("call_1", '{"temp": 22, "condition": "sunny"}'),
    ],
    tools=[
        ToolDef(
            name="get_weather",
            description="获取指定城市的天气信息",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        ),
    ],
    generation=GenerationConfig(temperature=0.7, max_output_tokens=1024),
)
```

### 快捷构造方法

`Message` 类提供了常用的工厂方法：

```python
# 纯文本 user 消息
msg = Message.user_text("Hello!")

# 纯文本 assistant 消息
msg = Message.assistant_text("Hi there!")

# 工具结果消息
msg = Message.tool_result("call_id", '{"result": "ok"}')

# 错误的工具结果
msg = Message.tool_result("call_id", "Error: not found", is_error=True)
```

## 场景三：网关调用 API

使用 `AnyLLMGateway` 实际调用 provider 的 API：

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
)

# 1. 初始化网关
gateway = AnyLLMGateway()

# 2. 注册 provider（适配器 + API 配置）
gateway.register_provider("openai_chat", ProviderConfig(
    adapter=OpenAIChatAdapter(),
    api_base="https://api.openai.com/v1",
    api_key="sk-xxx",
    timeout=60.0,
))

gateway.register_provider("anthropic", ProviderConfig(
    adapter=AnthropicAdapter(),
    api_base="https://api.anthropic.com",
    api_key="sk-ant-xxx",
))

# 3. 注册拦截器
gateway.register_interceptor(ImageResolutionInterceptor())
gateway.register_interceptor(RoleConsolidationInterceptor())

# 4. 构建请求并调用
request = UniversalRequest(
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    messages=[Message.user_text("Hello!")],
)

async def main():
    result = await gateway.chat_completions(request)

    # 输出内容
    for msg in result.response.output:
        for block in msg.content:
            if hasattr(block, 'text'):
                print(block.text)

    # Token 用量
    if result.response.usage:
        print(f"输入: {result.response.usage.input_tokens} tokens")
        print(f"输出: {result.response.usage.output_tokens} tokens")

    # 转换警告
    for w in result.warnings:
        print(f"[{w.severity}] {w.code}: {w.message}")

asyncio.run(main())
```

### 仅预览转换结果（不调用 API）

```python
result = asyncio.run(
    gateway.convert_only(request, target_provider="anthropic")
)
print(result.value)  # Anthropic 格式的请求 dict
```

## 场景四：自定义拦截器

三种方式创建自定义拦截器：

### 方式一：`@interceptor` 装饰器（最简洁）

```python
from anyllm import interceptor, TextBlock

@interceptor("add_watermark", only_for={"openai_chat", "anthropic"})
async def add_watermark(request, target_provider):
    """在系统指令中追加水印。"""
    request.instructions.append(TextBlock(text="Powered by AnyLLM"))
    return request

gateway.register_interceptor(add_watermark)
```

### 方式二：`FunctionInterceptor`（函数式）

```python
from anyllm import FunctionInterceptor

async def log_request(request, target_provider):
    print(f"[AnyLLM] -> {target_provider}: {len(request.messages)} msgs")
    return request

gateway.register_interceptor(
    FunctionInterceptor("logger", log_request)
)
```

### 方式三：继承 `BaseInterceptor`（完整控制）

```python
from anyllm import BaseInterceptor

class RateLimiter(BaseInterceptor):
    def __init__(self, max_rpm: int = 60):
        self._max_rpm = max_rpm
        self._call_count = 0

    @property
    def name(self) -> str:
        return "rate_limiter"

    async def process(self, request, target_provider):
        self._call_count += 1
        if self._call_count > self._max_rpm:
            raise RuntimeError("Rate limit exceeded")
        return request
```

## 下一步

- 深入了解 UIR 数据模型：[UIR 模型参考](uir-reference.md)
- 架构设计与数据流：[架构设计](architecture.md)
- 编写自己的适配器：[适配器开发指南](adapters.md)
- 拦截器机制详解：[拦截器开发指南](interceptors.md)
