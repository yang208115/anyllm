# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指引。

## 开发命令

- 安装开发依赖：
  - `pip install -e ".[dev]"`
- 代码检查（lint）：
  - `ruff check .`
- 代码格式化（如需要）：
  - `ruff format .`
- 运行验证脚本（当前测试/验收流程）：
  - `python validate_step1.py`
  - `python validate_step2_3.py`
  - `python validate_step4.py`
  - `python validate_step5.py`
- 按顺序运行全部验证脚本：
  - `python validate_step1.py && python validate_step2_3.py && python validate_step4.py && python validate_step5.py`
- 构建包：
  - `python -m build`

## 测试说明

- `pyproject.toml` 中已包含 pytest 配置（`testpaths = ["tests"]`），但当前仓库中暂无 `tests/` 目录。
- 现阶段请以 `validate_step*.py` 脚本作为主要集成验证。

## 高层架构

AnyLLM 是一个面向多 Provider 的 LLM API 转换与网关层。核心模式是：

- Provider 请求/响应 dict
- ↔ Adapter 层
- ↔ UIR（Universal Intermediate Representation）

### 主要分层

1. **Schema（UIR 模型）** — `anyllm/schema/`
   - 定义请求/响应/消息/content block/警告等统一数据模型。
   - 使用 Pydantic v2 的可辨别联合（基于 `type` 的多态）处理 block/tool/format 变体。

2. **Adapters** — `anyllm/adapters/`
   - `BaseAdapter` 定义四个方向的转换方法：
     - `request_to_uir`
     - `response_to_uir`
     - `uir_to_request`
     - `uir_to_response`
   - 当前已实现适配器：
     - `OpenAIChatAdapter`（`openai_chat.py`）
     - `AnthropicAdapter`（`anthropic.py`）

3. **转换编排层** — `anyllm/conversion/converter.py`
   - `UniversalConverter` 维护已注册适配器与拦截器管道。
   - 负责跨 Provider 请求转换链路：
     - source request → UIR → interceptors → target request
   - 返回带累计 warnings 的 `ConversionResult`。

4. **Interceptors（拦截器）** — `anyllm/interceptors.py`
   - 在目标转换前，于 UIR 请求层进行预处理。
   - 内置拦截器：
     - `ImageResolutionInterceptor`：为需要内联图片数据的 Provider（如 Anthropic/Bedrock 路径）将 URL 图片解析为 base64。
     - `RoleConsolidationInterceptor`：针对严格角色交替约束，执行 system/developer 提升及 tool 结果归并。
   - `FunctionInterceptor` 与 `@interceptor` 可用于快速实现轻量自定义中间件。

5. **网关运行时** — `anyllm/gateway.py`
   - `AnyLLMGateway` 组合路由、转换器与 HTTP 调用。
   - `ProviderConfig` 保存 adapter/API base/key/headers/timeout。
   - `chat_completions` 的执行流程：
     - 解析目标 provider（显式指定、自定义 router、或基于模型推断）
     - 执行拦截器
     - UIR → provider request
     - 使用 `httpx` 调用 provider API
     - provider response → UIR

6. **能力矩阵** — `anyllm/capabilities/matrix.py`
   - 按 provider profile 声明 L0/L1/L2 能力标记。
   - 供适配器/转换逻辑用于一致的降级处理与 warning 生成。

## 代码行为约定（重要）

- 转换应尽量保持语义；遇到不支持能力时，应发出 warning，而不是静默丢弃。
- `ConversionResult`（`anyllm/schema/warnings.py`）是 warnings 的统一载体；多步转换链路中应持续透传并累积。
- system/developer/tool 角色归一化应集中在拦截器层处理，避免在各 adapter 中重复实现。
- 网关当前按 provider 家族走 chat-completions 风格 HTTP 端点；URL/header 构造受 adapter 的 `provider_name` 影响。