# 拦截器开发指南

本文档介绍 AnyLLM 的拦截器（中间件）机制，包括内置拦截器的功能详解和自定义拦截器的三种创建方式。

## 概述

拦截器在 **UIR 层面** 对请求进行预处理，解决不同 provider 之间的格式兼容性问题。拦截器在适配器将 UIR 转换为目标 provider 格式之前执行。

```
UniversalRequest
      │
      ▼
Interceptor 1 (ImageResolutionInterceptor)
      │  下载 URL 图片 → base64
      ▼
Interceptor 2 (RoleConsolidationInterceptor)
      │  合并同角色消息，整理 tool result
      ▼
Interceptor 3 (自定义)
      │  业务自定义逻辑
      ▼
处理后的 UniversalRequest
      │
      ▼
Adapter.uir_to_request()  →  Provider dict
```

## 执行顺序

拦截器按注册顺序依次执行。推荐顺序：

1. `ImageResolutionInterceptor` — 先处理图片格式
2. `RoleConsolidationInterceptor` — 再整理消息角色和顺序
3. 自定义拦截器 — 按业务需求排列

```python
converter.register_interceptor(ImageResolutionInterceptor())      # 位置 0
converter.register_interceptor(RoleConsolidationInterceptor())    # 位置 1
converter.register_interceptor(my_interceptor)                    # 位置 2
```

可以通过 `position` 参数插入到指定位置：

```python
converter.register_interceptor(my_interceptor, position=0)  # 插入到最前面
```

## 内置拦截器

### ImageResolutionInterceptor

**功能**：自动将 URL 图片下载并转换为 base64 编码。

**触发条件**：目标 provider 是 Anthropic 或 Bedrock（这些 provider 不支持 URL 图片）。

**扫描范围**：

- `request.instructions` 中的 `ImageBlock`
- `request.messages[].content` 中的 `ImageBlock`
- `request.messages[].tool_results[].content` 中的 `ImageBlock`
- `ToolResultBlock` 内嵌的 `ImageBlock`

**配置参数**：

```python
interceptor = ImageResolutionInterceptor(
    timeout=30.0,           # 单张图片下载超时（秒）
    max_image_size=20*1024*1024,  # 最大图片大小（字节，默认 20MB）
)
```

**行为细节**：

| 条件 | 行为 |
|------|------|
| 目标是 OpenAI / Gemini / Ollama | 跳过（这些 provider 原生支持 URL） |
| source.kind 已是 base64 | 跳过 |
| 下载成功 | 就地更新 ImageBlock.source 为 base64 |
| 下载失败（超时/404） | 记录 warning 日志，保留原始 URL |
| 图片大小超过限制 | 记录 warning 日志，保留原始 URL |
| httpx 未安装 | 记录 warning 日志，跳过 |

**并发下载**：所有图片使用 `asyncio.gather` 并发下载，不会串行阻塞。

**Provider 图片支持矩阵**：

| Provider | URL | base64 | file_id | bytes |
|----------|-----|--------|---------|-------|
| OpenAI Chat | ✓ | ✓ | ✗ | ✗ |
| OpenAI Responses | ✓ | ✓ | ✓ | ✗ |
| Anthropic | ✗ | ✓ | ✗ | ✗ |
| Gemini | ✓ | ✓ | ✓ | ✗ |
| Bedrock | ✗ | ✗ | ✗ | ✓ |
| Ollama | ✓ | ✗ | ✗ | ✗ |

### RoleConsolidationInterceptor

**功能**：处理消息角色和顺序的兼容性问题。

**触发条件**：目标 provider 是 Anthropic、Google (Gemini) 或 Bedrock。

**处理步骤**：

#### 步骤 1：提升 system / developer 消息

将 `messages` 列表中的 `system` / `developer` 角色消息提升到 `instructions` 字段：

```
处理前：
  instructions: ["原有指令"]
  messages: [system("新指令"), user("Hello"), assistant("Hi")]

处理后：
  instructions: ["原有指令", "新指令"]
  messages: [user("Hello"), assistant("Hi")]
```

#### 步骤 2：合并 tool 消息到 user 消息

将 `tool` 角色消息中的 `ToolResult` 转移到相邻的 `user` 消息中（以 `ToolResultBlock` 形式嵌入 content）：

```
处理前（OpenAI 风格）：
  [assistant(tool_calls=[call_1, call_2]),
   tool(result_1),
   tool(result_2),
   user("好的")]

处理后（Anthropic 风格）：
  [assistant(tool_calls=[call_1, call_2]),
   user([tool_result_1, tool_result_2, "好的"])]
```

如果 tool 消息后面不是 user 消息，会自动创建一个新的 user 消息来容纳 tool results。

#### 步骤 3：合并连续同角色消息

Anthropic 和 Gemini 不允许连续出现两条相同角色的消息：

```
处理前：
  [user("A"), user("B"), assistant("C"), assistant("D")]

处理后：
  [user(["A", "B"]), assistant(["C", "D"])]
```

合并策略：
- 后一条消息的 `content` 追加到前一条末尾
- `tool_calls` 和 `tool_results` 也合并
- 后一条消息的 `id` / `name` / `provider` 字段被丢弃

**注意**：此拦截器会就地修改 `request.messages`。如果需要在后续操作中使用原始消息结构，请在调用前深拷贝。

## 创建自定义拦截器

### 方式一：@interceptor 装饰器

最简洁的方式，将一个 async 函数直接转换为拦截器：

```python
from anyllm import interceptor, TextBlock

@interceptor("add_disclaimer")
async def add_disclaimer(request, target_provider):
    """在系统指令中追加免责声明。"""
    request.instructions.append(
        TextBlock(text="注意：以上回答仅供参考。")
    )
    return request

# add_disclaimer 已经是一个 FunctionInterceptor 实例
converter.register_interceptor(add_disclaimer)
```

#### only_for 参数

限定拦截器只对特定 provider 生效：

```python
@interceptor("bedrock_fix", only_for={"bedrock"})
async def bedrock_fix(request, target_provider):
    """只对 Bedrock 执行的特殊处理。"""
    # Bedrock 特有的修复逻辑...
    return request
```

当 `target_provider` 不在 `only_for` 集合中时，拦截器自动跳过。

### 方式二：FunctionInterceptor

与装饰器方式等价，但更灵活（可动态创建）：

```python
from anyllm import FunctionInterceptor

async def log_request(request, target_provider):
    """记录请求日志。"""
    print(f"[AnyLLM] -> {target_provider}: {len(request.messages)} msgs")
    return request

logger_interceptor = FunctionInterceptor(
    "logger",
    log_request,
    only_for=None,  # 对所有 provider 生效
)

converter.register_interceptor(logger_interceptor)
```

### 方式三：继承 BaseInterceptor

适合需要维护状态或复杂初始化配置的场景：

```python
from anyllm.adapters.base import BaseInterceptor
from anyllm.schema.request import UniversalRequest

class TokenCounter(BaseInterceptor):
    """统计请求中的预估 token 数。"""

    def __init__(self, warn_threshold: int = 100000):
        self._warn_threshold = warn_threshold
        self._total_requests = 0

    @property
    def name(self) -> str:
        return "token_counter"

    async def process(
        self,
        request: UniversalRequest,
        target_provider: str,
    ) -> UniversalRequest:
        self._total_requests += 1

        total_chars = sum(
            len(block.text)
            for msg in request.messages
            for block in msg.content
            if hasattr(block, 'text')
        )

        est_tokens = total_chars // 4
        if est_tokens > self._warn_threshold:
            print(f"Warning: ~{est_tokens} tokens (threshold: {self._warn_threshold})")

        return request
```

## 函数签名

无论使用哪种方式，拦截器函数的签名都必须是：

```python
async def process(
    request: UniversalRequest,
    target_provider: str,
) -> UniversalRequest
```

| 参数 | 说明 |
|------|------|
| `request` | 当前的 UIR 请求对象（可就地修改） |
| `target_provider` | 目标 provider 标识符，如 `"anthropic"` |
| **返回值** | 处理后的 UniversalRequest（通常就是传入的同一个对象） |

## 拦截器管理

### 注册

```python
# 追加到末尾
converter.register_interceptor(my_interceptor)

# 插入到指定位置
converter.register_interceptor(my_interceptor, position=0)

# 同名拦截器自动替换
converter.register_interceptor(new_version_interceptor)
# 如果新旧 name 相同，旧的会被移除
```

### 注销

```python
# 按名称移除
removed = converter.unregister_interceptor("my_interceptor")
# removed = True 表示成功移除
```

### 查看

```python
# 查看所有已注册的拦截器名称（按执行顺序）
print(converter.registered_interceptors)
# ['image_resolution', 'role_consolidation', 'my_interceptor']
```

## 设计约束

1. **幂等性**：对同一请求多次执行应产生相同结果
2. **不修改 vendor**：`request.vendor` 是适配器的责任
3. **保持语义**：只做格式转换，不改变用户意图
4. **异步安全**：如需网络 I/O，必须用 async 实现
5. **容错**：拦截器执行失败不应阻塞请求流程（除非是致命错误）

## 完整示例：敏感词过滤拦截器

```python
import re
from anyllm import interceptor, TextBlock
from anyllm.schema.request import UniversalRequest

SENSITIVE_PATTERNS = [
    re.compile(r"密码\s*[:：]\s*\S+"),
    re.compile(r"api[_-]?key\s*[:=]\s*\S+", re.IGNORECASE),
]

@interceptor("sensitive_filter")
async def sensitive_filter(
    request: UniversalRequest,
    target_provider: str,
) -> UniversalRequest:
    """过滤请求中的敏感信息。"""
    for msg in request.messages:
        for block in msg.content:
            if isinstance(block, TextBlock):
                for pattern in SENSITIVE_PATTERNS:
                    block.text = pattern.sub("[REDACTED]", block.text)
    return request
```
