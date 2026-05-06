# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指引。

## 开发命令

- 安装开发依赖：
  - `uv sync"`
- 代码检查（lint）：
  - `uv run ruff check .`
- 代码格式化（如需要）：
  - `uv run ruff format .`
- 运行验证脚本（当前主验证流程）：
  - `uv run validate_step1.py`
  - `uv run validate_step2_3.py`
  - `uv run validate_step4.py`
  - `uv run validate_step5.py`
- 按顺序运行全部验证脚本：
  - `uv run validate_step1.py && uv run validate_step2_3.py && uv run validate_step4.py && uv run validate_step5.py`
- 构建包：
  - `uv run build`

## 测试说明

- `pyproject.toml` 中包含 pytest 配置（`testpaths = ["tests"]`），但当前仓库未提供 `tests/` 目录。
- 现阶段请以 `validate_step*.py` 作为主要集成验证入口。

## 高层架构

AnyLLM 是一个面向多 Provider 的 LLM API 转换与网关层。核心模式：

- Provider request/response dict
- ↔ Adapter 层
- ↔ UIR（Universal Intermediate Representation）

### 主要分层

1. **Schema（UIR 模型）** — `anyllm/schema/`
   - 定义统一请求、响应、消息、content block、tools、usage、warnings、stream event 等模型。
   - 基于 Pydantic v2 的可辨别联合（`type` discriminator）处理多态 block/format/tool choice。

2. **Adapters** — `anyllm/adapters/`
   - `BaseAdapter` 定义四个方向转换：
     - `request_to_uir`
     - `response_to_uir`
     - `uir_to_request`
     - `uir_to_response`
   - 当前内置适配器：
     - `OpenAIChatAdapter`（`openai_chat.py`）
     - `AnthropicAdapter`（`anthropic.py`）
     - `GeminiAdapter`（`gemini.py`，`provider_name="google"`）

3. **转换编排层** — `anyllm/conversion/converter.py`
   - `UniversalConverter` 管理 adapter 注册与 interceptor 管道。
   - 请求链路：source request → UIR → interceptors → target request。
   - 响应链路：source response → UIR → target response（无 interceptor）。
   - 流式链路：`stream_event_to_uir` 将 provider event 归一化为 `UniversalStreamEvent` 列表。

4. **Interceptors（拦截器）** — `anyllm/interceptors.py`
   - 在目标转换前，于 UIR 请求层执行预处理。
   - 内置拦截器：
     - `ImageResolutionInterceptor`：将 URL 图片解析为 base64（供目标 provider 使用）。
     - `RoleConsolidationInterceptor`：做 system/developer 提升与 tool 结果归并，满足严格角色约束。
   - 支持 `FunctionInterceptor` 与 `@interceptor` 快速构建轻量中间件。

5. **网关运行时** — `anyllm/gateway.py`
   - `AnyLLMGateway` 组合路由、转换器与 HTTP 调用。
   - `ProviderConfig` 保存 adapter / api_base / api_key / headers / timeout。
   - 核心能力：
     - `chat_completions`：完整非流式调用链路（路由 → 拦截器 → 请求转换 → HTTP → 响应转换）。
     - `convert_only`：仅做请求转换，不发起 HTTP。
     - `chat_completions_stream`：流式调用并产出 `UniversalStreamEvent`。
   - 路由要点：
     - 支持自定义 router。
     - 支持 provider alias（例如 `gemini` 会规范化为 `google`）。

6. **能力矩阵** — `anyllm/capabilities/matrix.py`
   - 按 provider profile 声明 L0/L1/L2 能力集。
   - 为适配器降级决策与 warning 生成提供一致依据。

## 当前行为约定（重要）

- 转换优先保持语义保真；不支持能力必须产出 warning，避免静默丢失。
- `ConversionResult`（`anyllm/schema/warnings.py`）是 warnings 的统一载体；多步链路中要持续透传并累积。
- system/developer/tool 的归一化应集中在拦截器层，而非在各 adapter 分散实现。
- 流式事件统一先归一化到 `UniversalStreamEvent`（`anyllm/schema/stream.py`），再由调用方决定后续处理。
- Google/Gemini 请求构造遵循网关实现：URL 中携带 model 与 key，payload 可去除重复 model 字段。
- 若新增 provider 或扩展 schema，需同步更新：
  - 对应 adapter
  - capability matrix
  - 验证脚本（`validate_step*.py`）与文档（`docs/`、`README.md`）

## 变更实施建议

- 新增或修改适配逻辑后，至少执行：
  - `ruff check .`
  - `python validate_step4.py`
  - `python validate_step5.py`
- 涉及基础 schema、拦截器、网关行为变更时，建议执行全量验证脚本。