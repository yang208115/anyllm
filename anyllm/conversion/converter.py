"""
UniversalConverter — 转换器总入口（PRD §13）。

这是 AnyLLM 的核心编排器，负责：
  1. 管理已注册的适配器（adapters）和拦截器（interceptors）
  2. 提供 Provider → UIR → Provider 的完整转换链路
  3. 在转换前后执行拦截器管道（Interceptor Pipeline）
  4. 合并多步转换产生的 warnings

典型使用流程：

    converter = UniversalConverter()

    # 注册适配器
    converter.register_adapter("openai_chat", OpenAIChatAdapter())
    converter.register_adapter("anthropic", AnthropicAdapter())

    # 注册拦截器（按执行顺序）
    converter.register_interceptor(ImageResolutionInterceptor())
    converter.register_interceptor(RoleConsolidationInterceptor())

    # 方式一：直接跨 provider 转换
    result = await converter.convert_request(
        source_provider="openai_chat",
        target_provider="anthropic",
        raw_request=openai_request_dict,
    )

    # 方式二：分步操作
    uir_result = converter.request_to_uir("openai_chat", openai_request_dict)
    processed_uir = await converter.run_interceptors(uir_result.value, "anthropic")
    target_result = converter.uir_to_request("anthropic", processed_uir)
"""

from __future__ import annotations

from typing import Any

from anyllm.adapters.base import BaseAdapter, BaseInterceptor
from anyllm.schema.request import UniversalRequest
from anyllm.schema.response import UniversalResponse
from anyllm.schema.stream import UniversalStreamEvent
from anyllm.schema.warnings import ConversionResult


class UniversalConverter:
    """
    通用转换器 — 管理适配器和拦截器，提供统一的转换 API。

    线程安全说明：
      适配器和拦截器实例本身是无状态的，UniversalConverter 也是无状态的，
      因此可以安全地在多个 asyncio task 之间共享同一个 converter 实例。
    """

    def __init__(self) -> None:
        # 已注册的适配器字典，键为 provider 标识符
        self._adapters: dict[str, BaseAdapter] = {}
        # 已注册的拦截器列表，按注册顺序执行
        self._interceptors: list[BaseInterceptor] = []

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register_adapter(self, provider: str, adapter: BaseAdapter) -> None:
        """
        注册一个厂商适配器。

        Args:
            provider: provider 标识符，如 "openai_chat"、"anthropic"。
                      同一个标识符重复注册会覆盖旧的适配器。
            adapter:  适配器实例。
        """
        self._adapters[provider] = adapter

    def register_interceptor(
        self,
        interceptor: BaseInterceptor,
        *,
        position: int | None = None,
    ) -> None:
        """
        注册一个拦截器。

        默认追加到管道末尾。如果指定了 position，则插入到指定位置。
        同名拦截器重复注册时会自动替换旧的（按 name 属性判断）。

        拦截器的执行顺序就是列表中的顺序，建议：
          1. ImageResolutionInterceptor — 先处理图片格式
          2. RoleConsolidationInterceptor — 再整理消息角色
          3. 自定义拦截器 — 按业务需求排列

        Args:
            interceptor: 拦截器实例（BaseInterceptor 子类或 FunctionInterceptor）。
            position:    可选，插入位置索引（0 = 最前面）。
                         为 None 时追加到末尾。
        """
        # 如果已有同名拦截器，先移除（避免重复）
        self._interceptors = [
            i for i in self._interceptors if i.name != interceptor.name
        ]

        if position is not None:
            self._interceptors.insert(position, interceptor)
        else:
            self._interceptors.append(interceptor)

    def unregister_interceptor(self, name: str) -> bool:
        """
        按名称移除已注册的拦截器。

        Args:
            name: 拦截器名称（BaseInterceptor.name 属性）。

        Returns:
            True 如果找到并移除了，False 如果未找到。
        """
        before = len(self._interceptors)
        self._interceptors = [
            i for i in self._interceptors if i.name != name
        ]
        return len(self._interceptors) < before

    @property
    def registered_interceptors(self) -> list[str]:
        """返回所有已注册的拦截器名称列表（按执行顺序）。"""
        return [i.name for i in self._interceptors]

    def get_adapter(self, provider: str) -> BaseAdapter:
        """
        获取已注册的适配器，未找到则抛出 KeyError。

        Args:
            provider: provider 标识符。

        Returns:
            对应的 BaseAdapter 实例。

        Raises:
            KeyError: 如果该 provider 尚未注册适配器。
        """
        if provider not in self._adapters:
            registered = list(self._adapters.keys())
            raise KeyError(
                f"未注册的 provider: '{provider}'，"
                f"已注册的 provider 列表: {registered}"
            )
        return self._adapters[provider]

    @property
    def registered_providers(self) -> list[str]:
        """返回所有已注册的 provider 标识符列表。"""
        return list(self._adapters.keys())

    # ------------------------------------------------------------------
    # 单向转换：Provider → UIR
    # ------------------------------------------------------------------

    def request_to_uir(
        self,
        provider: str,
        raw_request: dict[str, Any],
    ) -> ConversionResult[UniversalRequest]:
        """
        将 provider 原始请求转换为 UIR UniversalRequest。

        Args:
            provider:    来源 provider 标识符。
            raw_request: provider 原始请求 dict。

        Returns:
            ConversionResult[UniversalRequest]
        """
        adapter = self.get_adapter(provider)
        return adapter.request_to_uir(raw_request)

    def response_to_uir(
        self,
        provider: str,
        raw_response: dict[str, Any],
    ) -> ConversionResult[UniversalResponse]:
        """
        将 provider 原始响应转换为 UIR UniversalResponse。

        Args:
            provider:     来源 provider 标识符。
            raw_response: provider 原始响应 dict。

        Returns:
            ConversionResult[UniversalResponse]
        """
        adapter = self.get_adapter(provider)
        return adapter.response_to_uir(raw_response)

    # ------------------------------------------------------------------
    # 单向转换：UIR → Provider
    # ------------------------------------------------------------------

    def uir_to_request(
        self,
        provider: str,
        request: UniversalRequest,
    ) -> ConversionResult[dict[str, Any]]:
        """
        将 UIR UniversalRequest 转换为 provider 原始请求 dict。

        Args:
            provider: 目标 provider 标识符。
            request:  UIR UniversalRequest 对象。

        Returns:
            ConversionResult[dict]
        """
        adapter = self.get_adapter(provider)
        return adapter.uir_to_request(request)

    def uir_to_response(
        self,
        provider: str,
        response: UniversalResponse,
    ) -> ConversionResult[dict[str, Any]]:
        """
        将 UIR UniversalResponse 转换为 provider 原始响应 dict。

        Args:
            provider: 目标 provider 标识符。
            response: UIR UniversalResponse 对象。

        Returns:
            ConversionResult[dict]
        """
        adapter = self.get_adapter(provider)
        return adapter.uir_to_response(response)

    def stream_event_to_uir(
        self,
        provider: str,
        raw_event: dict[str, Any],
    ) -> ConversionResult[list[UniversalStreamEvent]]:
        """
        将 provider 原始流式事件转换为 UIR UniversalStreamEvent 列表。

        Args:
            provider: 来源 provider 标识符。
            raw_event: provider 流式事件对象。

        Returns:
            ConversionResult[list[UniversalStreamEvent]]
        """
        adapter = self.get_adapter(provider)
        return adapter.stream_provider_to_uir(raw_event)

    # ------------------------------------------------------------------
    # 拦截器管道
    # ------------------------------------------------------------------

    async def run_interceptors(
        self,
        request: UniversalRequest,
        target_provider: str,
    ) -> UniversalRequest:
        """
        按注册顺序依次执行所有拦截器。

        拦截器在「UIR 层面」对请求进行预处理，
        解决不同 provider 之间的格式兼容性问题。

        执行流程：
          request → Interceptor_1.process() → Interceptor_2.process() → ... → 处理后的 request

        Args:
            request:         待处理的 UIR 请求。
            target_provider: 目标 provider 标识符，拦截器可据此决定是否执行。

        Returns:
            处理后的 UniversalRequest（可能是原对象的就地修改，也可能是新对象）。
        """
        for interceptor in self._interceptors:
            request = await interceptor.process(request, target_provider)
        return request

    # ------------------------------------------------------------------
    # 完整转换链路：Provider A → UIR → (拦截器) → Provider B
    # ------------------------------------------------------------------

    async def convert_request(
        self,
        source_provider: str,
        target_provider: str,
        raw_request: dict[str, Any],
    ) -> ConversionResult[dict[str, Any]]:
        """
        完整的请求转换链路（PRD §13.1 convert_request）。

        流程：
          1. source adapter.request_to_uir(raw_request) → UIR
          2. 执行拦截器管道 → 处理后的 UIR
          3. target adapter.uir_to_request(UIR) → target dict
          4. 合并两步产生的 warnings

        Args:
            source_provider: 来源 provider 标识符。
            target_provider: 目标 provider 标识符。
            raw_request:     来源 provider 的原始请求 dict。

        Returns:
            ConversionResult[dict]
              .value    : 目标 provider 的请求 dict
              .warnings : 源转换 + 拦截器 + 目标转换的所有警告
        """
        # 步骤 1：源请求 → UIR
        source_result = self.request_to_uir(source_provider, raw_request)

        # 步骤 2：运行拦截器管道
        processed_request = await self.run_interceptors(
            source_result.value, target_provider
        )

        # 步骤 3：UIR → 目标请求
        target_result = self.uir_to_request(target_provider, processed_request)

        # 步骤 4：合并 warnings
        return ConversionResult(
            value=target_result.value,
            warnings=source_result.warnings + target_result.warnings,
        )

    async def convert_response(
        self,
        source_provider: str,
        target_provider: str,
        raw_response: dict[str, Any],
    ) -> ConversionResult[dict[str, Any]]:
        """
        完整的响应转换链路。

        流程：
          1. source adapter.response_to_uir(raw_response) → UIR Response
          2. target adapter.uir_to_response(UIR Response) → target dict
          3. 合并两步产生的 warnings

        注意：响应转换不经过拦截器管道（拦截器只处理请求）。

        Args:
            source_provider: 来源 provider 标识符。
            target_provider: 目标 provider 标识符。
            raw_response:    来源 provider 的原始响应 dict。

        Returns:
            ConversionResult[dict]
        """
        # 步骤 1：源响应 → UIR
        source_result = self.response_to_uir(source_provider, raw_response)

        # 步骤 2：UIR → 目标响应
        target_result = self.uir_to_response(target_provider, source_result.value)

        # 步骤 3：合并 warnings
        return ConversionResult(
            value=target_result.value,
            warnings=source_result.warnings + target_result.warnings,
        )
