"""
AnyLLMGateway — 网关入口（PRD §13 + 用户需求 Step 5）。

========================================================================
  设计概述
========================================================================

  AnyLLMGateway 是整个 AnyLLM 核心库的最高层入口，
  对外暴露统一的 chat_completions() 接口，对内编排：

    1. 适配器注册与路由（根据 model / provider 自动选择适配器）
    2. 拦截器管道（在 UIR 层面预处理请求）
    3. HTTP 客户端管理（调用 provider 的实际 API）
    4. 响应解析与统一化

  使用流程：

    ┌──────────────────────────────────────────────┐
    │  AnyLLMGateway                               │
    │                                              │
    │  AnyLLMRequest                               │
    │       │                                      │
    │       ▼                                      │
    │  UniversalConverter                           │
    │       │                                      │
    │       ├─ request_to_uir()  (如果输入是 dict) │
    │       │                                      │
    │       ├─ run_interceptors() (拦截器管道)      │
    │       │   ├─ ImageResolutionInterceptor       │
    │       │   └─ RoleConsolidationInterceptor     │
    │       │                                      │
    │       ├─ uir_to_request()  (UIR → 厂商 dict) │
    │       │                                      │
    │       ├─ HTTP 请求 → Provider API             │
    │       │                                      │
    │       ├─ response_to_uir() (厂商 dict → UIR)  │
    │       │                                      │
    │       ▼                                      │
    │  UniversalResponse                            │
    └──────────────────────────────────────────────┘

========================================================================
  网关 vs 转换器
========================================================================

  UniversalConverter : 纯格式转换，不涉及网络 I/O。
  AnyLLMGateway      : 在 Converter 之上增加了 HTTP 调用、API Key 管理、
                       provider 路由等「运行时」功能。

  对于只需要做格式转换（不调用 API）的场景，直接使用 UniversalConverter 即可。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from anyllm.adapters.base import BaseAdapter, BaseInterceptor
from anyllm.conversion.converter import UniversalConverter
from anyllm.schema.request import UniversalRequest
from anyllm.schema.response import UniversalResponse
from anyllm.schema.warnings import ConversionResult, ConversionWarning

logger = logging.getLogger("anyllm.gateway")


# =====================================================================
# Provider 配置
# =====================================================================

class ProviderConfig:
    """
    单个 provider 的运行时配置（API 地址、Key、适配器等）。

    使用示例：
        config = ProviderConfig(
            adapter=OpenAIChatAdapter(),
            api_base="https://api.openai.com/v1",
            api_key="sk-xxx",
        )
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        api_base: str | None = None,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Args:
            adapter         : 厂商适配器实例。
            api_base        : API 基础 URL，例如 "https://api.openai.com/v1"。
            api_key         : API Key，用于 Authorization header。
            default_headers : 额外的默认请求头。
            timeout         : HTTP 请求超时时间（秒）。
        """
        self.adapter = adapter
        self.api_base = api_base
        self.api_key = api_key
        self.default_headers = default_headers or {}
        self.timeout = timeout


# =====================================================================
# Gateway 响应
# =====================================================================

class GatewayResult:
    """
    网关处理结果，包含 UIR 响应和所有转换 warning。

    属性：
      response : 统一响应对象（UniversalResponse）。
      warnings : 整个请求链路中产生的所有警告。
      provider : 实际使用的 provider 标识符。
      raw_request  : 发给 provider 的原始请求 dict（用于调试）。
      raw_response : provider 返回的原始响应 dict（用于调试）。
    """

    def __init__(
        self,
        response: UniversalResponse,
        warnings: list[ConversionWarning],
        provider: str,
        raw_request: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        self.response = response
        self.warnings = warnings
        self.provider = provider
        self.raw_request = raw_request
        self.raw_response = raw_response

    @property
    def has_errors(self) -> bool:
        """是否包含 severity=error 级别的警告。"""
        return any(w.severity == "error" for w in self.warnings)


# =====================================================================
# AnyLLMGateway — 主入口
# =====================================================================

class AnyLLMGateway:
    """
    AnyLLM 网关 — 统一的大模型调用入口。

    核心功能：
      1. register_provider()     — 注册厂商配置（适配器 + API 配置）
      2. register_interceptor()  — 注册拦截器（中间件）
      3. chat_completions()      — 执行完整的请求链路
      4. convert_only()          — 只做格式转换，不调用 API

    使用示例：

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

        # 注册拦截器
        gateway.register_interceptor(ImageResolutionInterceptor())
        gateway.register_interceptor(RoleConsolidationInterceptor())

        # 调用
        request = UniversalRequest(
            model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
            messages=[Message.user_text("Hello!")],
        )
        result = await gateway.chat_completions(request)
        print(result.response.output[0].content[0].text)
    """

    def __init__(self) -> None:
        # provider 配置字典，键为 provider 标识符
        self._providers: dict[str, ProviderConfig] = {}
        # 内部转换器（管理适配器和拦截器）
        self._converter = UniversalConverter()
        # provider 路由策略（可自定义）
        self._router: Callable[[UniversalRequest], str] | None = None

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register_provider(
        self,
        name: str,
        config: ProviderConfig,
    ) -> None:
        """
        注册一个 provider（适配器 + API 配置）。

        Args:
            name   : provider 标识符，例如 "openai_chat", "anthropic"。
            config : ProviderConfig 实例，包含适配器和 API 配置。
        """
        self._providers[name] = config
        # 同步注册到内部转换器
        self._converter.register_adapter(name, config.adapter)
        logger.info("注册 provider: '%s' (adapter=%s)", name, type(config.adapter).__name__)

    def register_interceptor(
        self,
        interceptor_or_fn: BaseInterceptor,
        *,
        position: int | None = None,
    ) -> None:
        """
        注册一个拦截器。

        支持 BaseInterceptor 子类实例或 FunctionInterceptor。
        拦截器按注册顺序执行，可通过 position 参数指定插入位置。

        Args:
            interceptor_or_fn : 拦截器实例。
            position          : 可选，插入位置索引（0=最前）。
        """
        self._converter.register_interceptor(interceptor_or_fn, position=position)

    def set_router(
        self,
        router: Callable[[UniversalRequest], str],
    ) -> None:
        """
        设置自定义路由策略。

        默认路由策略：根据 request.model.provider 选择 provider。
        自定义路由可以实现更复杂的逻辑，例如负载均衡、故障转移、
        按模型名称路由到不同的 provider 等。

        Args:
            router: 路由函数，接收 UniversalRequest，
                    返回 provider 标识符（必须是已注册的 provider）。

        使用示例：
            def my_router(request):
                if "claude" in request.model.name:
                    return "anthropic"
                return "openai_chat"

            gateway.set_router(my_router)
        """
        self._router = router

    @property
    def registered_providers(self) -> list[str]:
        """返回所有已注册的 provider 标识符列表。"""
        return list(self._providers.keys())

    @property
    def registered_interceptors(self) -> list[str]:
        """返回所有已注册的拦截器名称列表（按执行顺序）。"""
        return self._converter.registered_interceptors

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def _canonicalize_provider(self, provider: str) -> str:
        provider_aliases = {
            "gemini": "google",
        }
        return provider_aliases.get(provider, provider)

    def _resolve_provider(self, request: UniversalRequest) -> str:
        """
        根据请求内容决定使用哪个 provider。

        路由优先级：
          1. 自定义路由函数（如果设置了 set_router）
          2. request.model.provider 字段
          3. 从 request.model.name 推断（模型名称匹配）

        Raises:
            ValueError: 如果无法确定 provider 或 provider 未注册。
        """
        # 优先级 1：自定义路由
        if self._router is not None:
            provider = self._router(request)
            if provider in self._providers:
                return provider
            raise ValueError(
                f"自定义路由返回了未注册的 provider: '{provider}'，"
                f"已注册: {list(self._providers.keys())}"
            )

        # 优先级 2：model.provider 字段
        model_provider = request.model.provider
        if model_provider != "unknown":
            canonical_provider = self._canonicalize_provider(model_provider)
            # 精确匹配
            if canonical_provider in self._providers:
                return canonical_provider
            # 模糊匹配（例如 "openai" 匹配 "openai_chat"）
            for name in self._providers:
                if name.startswith(canonical_provider):
                    return name

        # 优先级 3：从模型名称推断
        model_name = request.model.name.lower()
        provider_hints = {
            "gpt": "openai_chat",
            "o1": "openai_chat",
            "o3": "openai_chat",
            "claude": "anthropic",
            "gemini": "google",
            "llama": "ollama",
            "mistral": "ollama",
            "qwen": "ollama",
        }
        for hint, provider in provider_hints.items():
            if hint in model_name and provider in self._providers:
                return provider

        # 兜底：如果只注册了一个 provider，直接使用
        if len(self._providers) == 1:
            return next(iter(self._providers))

        raise ValueError(
            f"无法为模型 '{request.model.name}' (provider='{model_provider}') "
            f"确定目标 provider。已注册: {list(self._providers.keys())}。"
            f"请设置 request.model.provider 或使用 set_router()。"
        )

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    async def chat_completions(
        self,
        request: UniversalRequest,
        *,
        provider: str | None = None,
        http_client: Any | None = None,
    ) -> GatewayResult:
        """
        执行完整的聊天补全请求链路。

        流程：
          1. 路由：确定目标 provider
          2. 拦截器管道：预处理请求（图片转换、角色整理等）
          3. 适配器转换：UIR → provider 请求 dict
          4. HTTP 请求：调用 provider API
          5. 响应解析：provider 响应 dict → UIR

        Args:
            request     : UIR 统一请求对象。
            provider    : 可选，强制指定 provider（跳过路由）。
            http_client : 可选，自定义 HTTP 客户端（httpx.AsyncClient）。
                          为 None 时自动创建。

        Returns:
            GatewayResult，包含 UIR 响应和所有警告。
        """
        all_warnings: list[ConversionWarning] = []

        # ---- 步骤 1：路由 ----
        target = self._canonicalize_provider(provider) if provider else self._resolve_provider(request)
        config = self._providers[target]
        logger.info("路由到 provider: '%s' (model=%s)", target, request.model.name)

        # ---- 步骤 2：拦截器管道 ----
        processed = await self._converter.run_interceptors(request, target)

        # ---- 步骤 3：UIR → provider 请求 ----
        convert_result = self._converter.uir_to_request(target, processed)
        all_warnings.extend(convert_result.warnings)
        raw_request = convert_result.value

        # ---- 步骤 4：HTTP 请求 ----
        raw_response = await self._call_provider_api(
            config, raw_request, http_client
        )

        # ---- 步骤 5：provider 响应 → UIR ----
        response_result = config.adapter.response_to_uir(raw_response)
        all_warnings.extend(response_result.warnings)

        return GatewayResult(
            response=response_result.value,
            warnings=all_warnings,
            provider=target,
            raw_request=raw_request,
            raw_response=raw_response,
        )

    async def convert_only(
        self,
        request: UniversalRequest,
        *,
        target_provider: str | None = None,
    ) -> ConversionResult[dict[str, Any]]:
        """
        只做格式转换（UIR → provider dict），不调用 API。

        适用场景：
          - 预览转换结果
          - 与自己的 HTTP 客户端集成
          - 测试和调试转换逻辑

        Args:
            request         : UIR 统一请求对象。
            target_provider : 可选，目标 provider。为 None 时自动路由。

        Returns:
            ConversionResult[dict]，包含 provider 请求 dict 和所有警告。
        """
        target = self._canonicalize_provider(target_provider) if target_provider else self._resolve_provider(request)

        # 执行拦截器管道
        processed = await self._converter.run_interceptors(request, target)

        # UIR → provider 请求
        return self._converter.uir_to_request(target, processed)

    # ------------------------------------------------------------------
    # HTTP 调用
    # ------------------------------------------------------------------

    async def _call_provider_api(
        self,
        config: ProviderConfig,
        raw_request: dict[str, Any],
        http_client: Any | None = None,
    ) -> dict[str, Any]:
        """
        调用 provider 的 HTTP API。

        根据 provider 类型构造不同的 URL 和 headers：
          OpenAI     : POST {api_base}/chat/completions
                       Authorization: Bearer {api_key}
          Anthropic  : POST {api_base}/v1/messages
                       x-api-key: {api_key}
                       anthropic-version: 2023-06-01

        Args:
            config      : provider 配置。
            raw_request : 已转换的 provider 请求 dict。
            http_client : 可选，外部传入的 httpx.AsyncClient。

        Returns:
            provider 原始响应 dict。

        Raises:
            ImportError: httpx 未安装。
            httpx.HTTPStatusError: API 返回非 2xx 状态码。
        """
        try:
            import httpx
        except ImportError as err:
            raise ImportError(
                "httpx 是调用 provider API 的必要依赖，"
                "请安装: pip install httpx"
            ) from err

        # 构造请求 URL 和 headers
        url, headers = self._build_request_params(config, raw_request)

        payload = raw_request
        if "google" in config.adapter.provider_name and "model" in raw_request:
            payload = {k: v for k, v in raw_request.items() if k != "model"}

        # 使用外部客户端或创建新的
        if http_client is not None:
            response = await http_client.post(
                url, json=payload, headers=headers
            )
        else:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    url, json=payload, headers=headers
                )

        response.raise_for_status()
        return response.json()

    def _build_request_params(
        self,
        config: ProviderConfig,
        raw_request: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        """
        根据 provider 配置构造请求 URL 和 headers。

        Returns:
            (url: str, headers: dict)
        """
        provider_name = config.adapter.provider_name
        headers = {"Content-Type": "application/json"}
        headers.update(config.default_headers)

        if "anthropic" in provider_name:
            # Anthropic API
            base = (config.api_base or "https://api.anthropic.com").rstrip("/")
            url = f"{base}/v1/messages"
            if config.api_key:
                headers["x-api-key"] = config.api_key
            headers["anthropic-version"] = "2023-06-01"

        elif "openai" in provider_name or "ollama" in provider_name:
            # OpenAI / OpenAI-compatible API
            base = (config.api_base or "https://api.openai.com/v1").rstrip("/")
            # 如果 base 已经包含 /v1，不重复添加
            if base.endswith("/v1"):
                url = f"{base}/chat/completions"
            else:
                url = f"{base}/v1/chat/completions"
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

        elif "google" in provider_name:
            # Gemini API
            base = (config.api_base or "https://generativelanguage.googleapis.com").rstrip("/")
            model = raw_request.get("model") or "gemini-1.5-pro"
            url = f"{base}/v1beta/models/{model}:generateContent"
            if config.api_key:
                parsed = urlsplit(url)
                query = dict(parse_qsl(parsed.query, keep_blank_values=True))
                query["key"] = config.api_key
                url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))

        else:
            # 通用兜底
            base = (config.api_base or "http://localhost:11434").rstrip("/")
            url = f"{base}/v1/chat/completions"
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

        return url, headers
