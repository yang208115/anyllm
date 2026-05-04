"""
适配器基类 — 所有厂商适配器的抽象接口（PRD §4, §12）。

========================================================================
  设计思路
========================================================================

  Provider Request ──► request_to_uir() ──► UniversalRequest (UIR)
                                                │
  Provider Response ◄── uir_to_request() ◄──────┘
                                                │
  Provider Response ──► response_to_uir() ──► UniversalResponse (UIR)
                                                │
                        uir_to_response() ◄─────┘

每个 provider 至少实现 4 个方法：
  1. request_to_uir   : 将 provider 原始请求 dict → UIR UniversalRequest
  2. response_to_uir  : 将 provider 原始响应 dict → UIR UniversalResponse
  3. uir_to_request   : 将 UIR UniversalRequest → provider 请求 dict
  4. uir_to_response  : 将 UIR UniversalResponse → provider 响应 dict

所有方法都必须返回 ConversionResult[T]，而不是裸返回 dict / 模型对象，
以便调用方（UniversalConverter / Gateway）统一收集和透传 warnings。

========================================================================
  能力声明 (ProviderCapabilities)
========================================================================

适配器通过 capabilities 属性声明自身支持的能力等级（PRD §12），
使得 UniversalConverter / Gateway 在转换前就能检测到不兼容的特性，
提前发出有意义的 warning，而不是等 provider 返回 400 错误。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict

from anyllm.schema.request import UniversalRequest
from anyllm.schema.response import UniversalResponse
from anyllm.schema.warnings import ConversionResult


# =====================================================================
# Provider 能力矩阵 (PRD §12)
# =====================================================================

class ProviderCapabilities(BaseModel):
    """
    声明某个 provider / 适配器支持的功能集合。

    用途：
      1. 在转换前快速判断源请求是否能被目标 provider 处理。
      2. 如果目标不支持某能力（如 image_input=False），
         UniversalConverter 可以提前插入 UNSUPPORTED_MODALITY warning。
      3. 用于 Gateway 的智能路由——优先选择能力覆盖最全的 provider。

    能力层级对应 PRD §11：
      L0 (所有 provider): text, system_instruction, streaming
      L1 (大部分 provider): image_input, tools, json_object/json_schema
      L2 (部分 provider): stateful, builtin_tools, developer_role
    """

    model_config = ConfigDict(populate_by_name=True)

    # ---------- L0: 纯文本聊天 ----------
    text: bool = True
    """是否支持纯文本聊天——所有 provider 都应该为 True。"""

    system_instruction: bool = True
    """是否支持 system / instructions 指令，几乎所有 provider 都支持。"""

    streaming: bool = True
    """是否支持流式响应（SSE / chunked transfer）。"""

    # ---------- L1: 多模态 + 工具调用 ----------
    image_input: bool = False
    """是否支持图片输入（user 消息中的 ImageBlock）。"""

    audio_input: bool = False
    """是否支持音频输入（user 消息中的 AudioBlock）。"""

    file_input: bool = False
    """是否支持文件输入（user 消息中的 FileBlock）。"""

    tools: bool = False
    """是否支持 function calling / tool use。"""

    parallel_tool_calls: bool = False
    """是否支持单次响应中返回多个 tool_calls（并行调用）。"""

    tool_result_blocks: bool = False
    """
    是否原生支持 ToolResult 以 content block 形式嵌入 user 消息。
    True : Anthropic / Gemini / Bedrock（content 里的 tool_result）
    False: OpenAI Chat（独立的 tool 角色消息）
    """

    json_object: bool = False
    """是否支持 json_object 格式的结构化输出。"""

    json_schema: bool = False
    """是否支持 json_schema（Structured Outputs）格式的结构化输出。"""

    # ---------- L2: 高级 agent / provider 特性 ----------
    stateful: bool = False
    """是否支持有状态会话（服务端保留对话历史）。"""

    previous_response_id: bool = False
    """是否支持 previous_response_id 字段（OpenAI Responses API 独有）。"""

    builtin_tools: bool = False
    """是否支持 provider 内置工具（web_search、file_search、code_interpreter 等）。"""

    developer_role: bool = False
    """是否支持 developer 角色消息（OpenAI Responses API 独有）。"""


# =====================================================================
# 适配器抽象基类 (PRD §4)
# =====================================================================

class BaseAdapter(ABC):
    """
    厂商适配器抽象基类。

    所有适配器必须继承此类并实现 4 个核心转换方法。
    适配器实例是无状态的，同一个实例可以安全地在多个请求间复用。

    实现规范：
      - 所有方法返回 ConversionResult，不要直接返回 dict 或模型对象。
      - 转换过程中遇到不支持的特性时，应该发出 warning 而不是抛出异常。
        只有当数据格式严重损坏导致无法继续转换时，才应抛出异常。
      - 厂商特有字段放入 vendor / provider 透传区，不要静默丢弃。

    使用示例：
        adapter = OpenAIChatAdapter()
        result = adapter.uir_to_request(uir_request)
        if result.has_errors:
            logger.error("转换失败: %s", result.warnings)
        else:
            provider_dict = result.value
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        适配器对应的 provider 标识符。

        返回值应与 ProviderName 枚举或注册键保持一致，
        例如 "openai_chat", "anthropic", "google", "bedrock", "ollama"。
        此标识符用于：
          - UniversalConverter 的 adapters 字典键名
          - ConversionWarning 中标注来源
          - vendor 透传字段的命名空间
        """
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """
        声明此适配器 / provider 支持的能力集合。

        子类应覆盖此属性以反映真实能力。
        默认返回最保守的配置（仅文本 + 流式），避免误报。
        """
        return ProviderCapabilities()

    # ------------------------------------------------------------------
    # 核心转换方法
    # ------------------------------------------------------------------

    @abstractmethod
    def request_to_uir(
        self,
        raw_request: Dict[str, Any],
    ) -> ConversionResult[UniversalRequest]:
        """
        将 provider 原始请求（dict）转换为 UIR UniversalRequest。

        转换流程（以 OpenAI Chat 为例）：
          1. 提取 messages、tools、model 等字段
          2. 将 system role 消息转为 instructions
          3. 将 tool_calls 解析为 ToolCall 对象（JSON string → dict）
          4. 将 image_url 内容转为 ImageBlock + MediaSource
          5. 无法识别的字段包装到 vendor / ProviderBlock 中

        Args:
            raw_request: provider 原始请求 dict，
                         结构取决于具体 provider（OpenAI/Anthropic/...）。

        Returns:
            ConversionResult[UniversalRequest]
              .value    : 转换后的 UniversalRequest 对象
              .warnings : 转换过程中产生的警告列表
        """
        ...

    @abstractmethod
    def response_to_uir(
        self,
        raw_response: Dict[str, Any],
    ) -> ConversionResult[UniversalResponse]:
        """
        将 provider 原始响应（dict）转换为 UIR UniversalResponse。

        转换流程（以 OpenAI Chat 为例）：
          1. 提取 choices[0].message 作为 output Message
          2. 将 finish_reason 映射为 StopReason
          3. 提取 usage 信息
          4. 将 tool_calls 解析为 ToolCall 对象

        Args:
            raw_response: provider 原始响应 dict。

        Returns:
            ConversionResult[UniversalResponse]
              .value    : 转换后的 UniversalResponse 对象
              .warnings : 转换过程中产生的警告列表
        """
        ...

    @abstractmethod
    def uir_to_request(
        self,
        request: UniversalRequest,
    ) -> ConversionResult[Dict[str, Any]]:
        """
        将 UIR UniversalRequest 转换为 provider 原始请求（dict）。

        这是「翻译」的核心——需要处理各种格式差异，例如：
          - system prompt 的放置位置（顶层 vs 首条消息 vs systemInstruction）
          - tool_calls 的表达方式（顶层列表 vs content block）
          - image 的传递方式（url vs base64 vs bytes）
          - 各种字段名映射（max_tokens vs maxTokens vs max_output_tokens）

        Args:
            request: UIR UniversalRequest 对象。

        Returns:
            ConversionResult[dict]
              .value    : 目标 provider 接受的请求 dict
              .warnings : 转换过程中产生的降级/透传警告
        """
        ...

    @abstractmethod
    def uir_to_response(
        self,
        response: UniversalResponse,
    ) -> ConversionResult[Dict[str, Any]]:
        """
        将 UIR UniversalResponse 转换为 provider 原始响应（dict）。

        使用场景较少（通常是代理网关需要把内部响应格式转回客户端期望的格式），
        但对于完整的双向转换链路来说是必需的。

        Args:
            response: UIR UniversalResponse 对象。

        Returns:
            ConversionResult[dict]
              .value    : 目标 provider 格式的响应 dict
              .warnings : 转换过程中产生的警告
        """
        ...


# =====================================================================
# 拦截器 / 中间件抽象基类
# =====================================================================

class BaseInterceptor(ABC):
    """
    请求拦截器（中间件）抽象基类。

    拦截器在 UIR 层面对请求进行预处理或后处理，
    解决不同 provider 之间的兼容性问题。

    ========================================================================
      执行时机与顺序
    ========================================================================

    请求到达 Gateway 后、发送给适配器之前，按注册顺序依次执行：

      UniversalRequest
        │
        ▼
      Interceptor 1 (例: ImageResolutionInterceptor)
        │ 下载 URL 图片，转为 base64
        ▼
      Interceptor 2 (例: RoleConsolidationInterceptor)
        │ 合并连续同角色消息，整理 tool result 位置
        ▼
      处理后的 UniversalRequest → 交给 Adapter.uir_to_request()

    ========================================================================
      设计约束
    ========================================================================

    1. 拦截器必须是 **幂等** 的：对同一请求多次执行应产生相同结果。
    2. 拦截器不应修改 request.vendor 中的内容（那是适配器的责任）。
    3. 拦截器可以修改 messages、instructions、tools，
       但应尽量保持语义不变（只做格式转换，不改变用户意图）。
    4. 如果拦截器需要做网络 I/O（如下载图片），必须用 async 实现。

    使用示例：
        interceptor = ImageResolutionInterceptor()
        result = await interceptor.process(request, target_provider="anthropic")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        拦截器名称，用于日志和调试。

        建议使用 snake_case，例如 "image_resolution", "role_consolidation"。
        """
        ...

    @abstractmethod
    async def process(
        self,
        request: UniversalRequest,
        target_provider: str,
    ) -> UniversalRequest:
        """
        对 UIR 请求进行预处理。

        参数说明：
          request         : 当前的 UniversalRequest，可直接修改并返回。
                            （注意：Pydantic 模型默认可变，调用方不需要深拷贝。）
          target_provider : 目标 provider 标识符，例如 "anthropic"。
                            拦截器可以根据目标 provider 决定是否需要执行转换。
                            例如 ImageResolutionInterceptor 只有在目标是 Anthropic/Bedrock
                            时才需要将 URL 图片转为 base64。

        返回值：
          处理后的 UniversalRequest（通常就是传入的同一个对象，直接修改后返回）。

        异常处理：
          如果拦截器执行失败（如网络超时），应记录日志但不阻塞请求流程，
          具体策略由子类实现决定。
        """
        ...
