"""
内置拦截器 (Interceptors) — 解决跨 provider 的格式兼容性问题。

========================================================================
  拦截器概览
========================================================================

  1. ImageResolutionInterceptor
     ▸ 将 URL 图片自动下载并转换为 base64 编码。
     ▸ 解决 Anthropic / Bedrock 不支持 URL 图片的问题。
     ▸ 需要异步 HTTP 请求（使用 httpx）。

  2. RoleConsolidationInterceptor
     ▸ 合并连续的同角色消息，避免 provider 拒绝请求。
     ▸ 将 tool 角色消息中的 ToolResult 转移到 user 角色消息（Anthropic 规则）。
     ▸ 确保 Anthropic 的「user/assistant 必须交替出现」约束被满足。
     ▸ 纯同步操作，不涉及网络 I/O。

  3. FunctionInterceptor
     ▸ 轻量级拦截器，允许用户通过一个普通的 async 函数创建自定义拦截器，
       不需要编写完整的类。适合简单的请求修改场景。

========================================================================
  自定义拦截器注册方式
========================================================================

  方式一：继承 BaseInterceptor（完整控制）

    class MyInterceptor(BaseInterceptor):
        @property
        def name(self) -> str:
            return "my_interceptor"

        async def process(self, request, target_provider):
            # 修改 request ...
            return request

  方式二：使用 FunctionInterceptor（快速创建）

    async def add_disclaimer(request, target_provider):
        request.instructions.append(TextBlock(text="免责声明..."))
        return request

    interceptor = FunctionInterceptor("add_disclaimer", add_disclaimer)

  方式三：使用 @interceptor 装饰器（最简洁）

    @interceptor("add_disclaimer")
    async def add_disclaimer(request, target_provider):
        request.instructions.append(TextBlock(text="免责声明..."))
        return request

    # add_disclaimer 现在是一个 FunctionInterceptor 实例

========================================================================
  执行顺序建议
========================================================================

  推荐注册顺序：
    1. ImageResolutionInterceptor  （先处理图片，避免后续操作丢失图片信息）
    2. RoleConsolidationInterceptor（最后整理消息结构）
    3. 自定义拦截器按业务需求自行排序

========================================================================
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from copy import deepcopy
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

from anyllm.adapters.base import BaseInterceptor
from anyllm.schema.content import (
    ContentBlock,
    ImageBlock,
    MediaSource,
    TextBlock,
    ToolCallBlock,
    ToolResult,
    ToolResultBlock,
)
from anyllm.schema.message import Message
from anyllm.schema.request import UniversalRequest

logger = logging.getLogger("anyllm.interceptors")

# 自定义拦截器函数的类型签名：
#   (request: UniversalRequest, target_provider: str) -> UniversalRequest
InterceptorFn = Callable[
    [UniversalRequest, str],
    Awaitable[UniversalRequest],
]


# =====================================================================
# FunctionInterceptor — 用普通函数快速创建自定义拦截器
# =====================================================================

class FunctionInterceptor(BaseInterceptor):
    """
    函数式拦截器 — 将一个普通的 async 函数包装为 BaseInterceptor 实例。

    适合不需要维护状态、逻辑较简单的场景。
    对于需要初始化配置、维护状态的复杂拦截器，建议直接继承 BaseInterceptor。

    使用示例：

        async def add_watermark(request, target_provider):
            \"\"\"在每条 user 消息末尾追加水印文本。\"\"\"
            for msg in request.messages:
                if msg.role == "user":
                    msg.content.append(TextBlock(text="[via AnyLLM]"))
            return request

        # 创建拦截器实例
        watermark = FunctionInterceptor("add_watermark", add_watermark)

        # 注册到转换器
        converter.register_interceptor(watermark)

    参数说明：
        interceptor_name : 拦截器名称，用于日志和调试输出。
        fn               : 异步拦截函数，签名必须为：
                           async def(request: UniversalRequest, target_provider: str)
                               -> UniversalRequest
        only_for         : 可选，限定只对指定的目标 provider 执行。
                           例如 {"anthropic", "bedrock"} 表示只对这两个 provider 生效。
                           为 None 时对所有 provider 生效。
    """

    def __init__(
        self,
        interceptor_name: str,
        fn: InterceptorFn,
        *,
        only_for: Optional[Set[str]] = None,
    ) -> None:
        self._name = interceptor_name
        self._fn = fn
        self._only_for = only_for

    @property
    def name(self) -> str:
        return self._name

    async def process(
        self,
        request: UniversalRequest,
        target_provider: str,
    ) -> UniversalRequest:
        """
        如果 only_for 限制了目标 provider，且当前目标不在范围内，则跳过执行。
        否则调用包装的函数。
        """
        # 检查是否需要跳过
        if self._only_for is not None and target_provider not in self._only_for:
            logger.debug(
                "拦截器 '%s' 不适用于目标 provider '%s'，跳过。",
                self._name, target_provider,
            )
            return request

        logger.debug("执行自定义拦截器 '%s'（目标: %s）", self._name, target_provider)
        return await self._fn(request, target_provider)


def interceptor(
    name: str,
    *,
    only_for: Optional[Set[str]] = None,
) -> Callable[[InterceptorFn], FunctionInterceptor]:
    """
    装饰器 — 将一个 async 函数转换为 FunctionInterceptor 实例。

    这是创建自定义拦截器的最简洁方式。

    使用示例：

        @interceptor("add_disclaimer")
        async def add_disclaimer(request, target_provider):
            from anyllm.schema.content import TextBlock
            request.instructions.append(TextBlock(text="Disclaimer: ..."))
            return request

        # add_disclaimer 现在是一个 FunctionInterceptor 实例，可以直接注册：
        converter.register_interceptor(add_disclaimer)

    带 only_for 限制：

        @interceptor("bedrock_fix", only_for={"bedrock"})
        async def bedrock_fix(request, target_provider):
            # 只对 Bedrock 生效的修复逻辑
            ...
            return request

    Args:
        name     : 拦截器名称。
        only_for : 可选，限定只对指定目标 provider 生效的集合。

    Returns:
        装饰器函数，将 async 函数转换为 FunctionInterceptor 实例。
    """
    def decorator(fn: InterceptorFn) -> FunctionInterceptor:
        return FunctionInterceptor(name, fn, only_for=only_for)
    return decorator


# =====================================================================
# 内置常量
# =====================================================================

# 需要将 URL 图片预转换为 base64 的 provider 集合
# 这些 provider 不支持直接传递图片 URL
_PROVIDERS_REQUIRING_BASE64_IMAGE: Set[str] = {
    "anthropic",
    "bedrock",
}


# =====================================================================
# ImageResolutionInterceptor
# =====================================================================

class ImageResolutionInterceptor(BaseInterceptor):
    """
    图片格式自动转换拦截器。

    ========================================================================
      解决的问题
    ========================================================================

    不同 provider 对图片来源的支持差异：
      ┌──────────────┬───────┬────────┬─────────┬───────┐
      │ Provider     │  URL  │ base64 │ file_id │ bytes │
      ├──────────────┼───────┼────────┼─────────┼───────┤
      │ OpenAI Chat  │  ✓    │   ✓    │   ✗     │  ✗    │
      │ OpenAI Resp  │  ✓    │   ✓    │   ✓     │  ✗    │
      │ Anthropic    │  ✗    │   ✓    │   ✗     │  ✗    │
      │ Gemini       │  ✓    │   ✓    │   ✓     │  ✗    │
      │ Bedrock      │  ✗    │   ✗    │   ✗     │  ✓    │
      │ Ollama       │  ✓    │   ✗    │   ✗     │  ✗    │
      └──────────────┴───────┴────────┴─────────┴───────┘

    当目标 provider 是 Anthropic 或 Bedrock 时，
    本拦截器会自动下载 URL 图片并转换为 base64 编码。

    ========================================================================
      转换逻辑
    ========================================================================

    对请求中所有 ImageBlock 执行以下检查：
      1. 如果 source.kind == "url" 且目标 provider 需要 base64：
         → 用 httpx 下载图片 → 转为 base64 → 更新 source
      2. 如果 source.kind 已经是 "base64"：
         → 不做任何处理
      3. 如果下载失败（超时、404 等）：
         → 记录日志，保留原始 URL（让适配器层处理降级）

    ========================================================================
      性能考虑
    ========================================================================

    - 所有图片下载并发执行（使用 asyncio.gather）。
    - 设置合理的超时时间（默认 30 秒）。
    - 限制最大图片大小（默认 20MB），超出则跳过转换。
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_image_size: int = 20 * 1024 * 1024,
    ) -> None:
        """
        Args:
            timeout        : 单张图片下载超时时间（秒）。
            max_image_size : 允许的最大图片字节数，超出则跳过转换（默认 20MB）。
        """
        self._timeout = timeout
        self._max_image_size = max_image_size

    @property
    def name(self) -> str:
        return "image_resolution"

    async def process(
        self,
        request: UniversalRequest,
        target_provider: str,
    ) -> UniversalRequest:
        """
        扫描请求中的所有图片块，根据目标 provider 需要进行格式转换。

        处理范围：
          - request.instructions 中的 ImageBlock
          - request.messages[].content 中的 ImageBlock
          - request.messages[].tool_results[].content 中的 ImageBlock
            （ToolResult 富文本结果中也可能包含图片）
        """
        # 如果目标 provider 不需要 base64 转换，直接跳过
        if target_provider not in _PROVIDERS_REQUIRING_BASE64_IMAGE:
            logger.debug(
                "目标 provider '%s' 支持 URL 图片，跳过图片转换。",
                target_provider,
            )
            return request

        logger.info(
            "目标 provider '%s' 需要 base64 图片，开始扫描并转换...",
            target_provider,
        )

        # 收集所有需要转换的 ImageBlock 引用
        image_blocks: List[ImageBlock] = []

        # 扫描 instructions
        for block in request.instructions:
            if isinstance(block, ImageBlock) and block.source.kind == "url":
                image_blocks.append(block)

        # 扫描 messages
        for msg in request.messages:
            # 扫描 content
            for block in msg.content:
                if isinstance(block, ImageBlock) and block.source.kind == "url":
                    image_blocks.append(block)
                # 扫描 ToolResultBlock 内嵌的图片
                elif isinstance(block, ToolResultBlock):
                    for inner in block.result.content:
                        if isinstance(inner, ImageBlock) and inner.source.kind == "url":
                            image_blocks.append(inner)

            # 扫描 tool_results 顶层列表中的图片
            for result in msg.tool_results:
                for inner in result.content:
                    if isinstance(inner, ImageBlock) and inner.source.kind == "url":
                        image_blocks.append(inner)

        if not image_blocks:
            logger.debug("请求中没有需要转换的 URL 图片。")
            return request

        logger.info("找到 %d 张 URL 图片需要转换为 base64。", len(image_blocks))

        # 并发下载所有图片
        import asyncio
        tasks = [self._resolve_image(block) for block in image_blocks]
        await asyncio.gather(*tasks)

        return request

    async def _resolve_image(self, block: ImageBlock) -> None:
        """
        下载单张 URL 图片，将其就地转换为 base64 编码。

        转换结果直接修改 block.source（Pydantic 模型默认可变），
        无需返回新对象。

        如果下载失败，记录警告日志但不抛出异常，保留原始 URL。
        """
        url = block.source.value
        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                # 检查图片大小
                content_length = len(response.content)
                if content_length > self._max_image_size:
                    logger.warning(
                        "图片过大 (%d bytes > %d bytes)，跳过转换: %s",
                        content_length, self._max_image_size, url,
                    )
                    return

                # 确定 MIME 类型
                # 优先使用 HTTP 响应头中的 Content-Type
                content_type = response.headers.get("content-type", "")
                mime_type = content_type.split(";")[0].strip() if content_type else None

                # 如果响应头中没有，尝试从 URL 路径推断
                if not mime_type or mime_type == "application/octet-stream":
                    guessed, _ = mimetypes.guess_type(url)
                    mime_type = guessed or "image/png"

                # 编码为 base64
                b64_data = base64.b64encode(response.content).decode("ascii")

                # 就地更新 ImageBlock.source
                block.source = MediaSource(
                    kind="base64",
                    value=b64_data,
                    mime_type=mime_type,
                )

                logger.debug(
                    "成功将 URL 图片转为 base64: %s (%s, %d bytes)",
                    url, mime_type, content_length,
                )

        except ImportError:
            # httpx 未安装，记录警告并跳过
            logger.warning(
                "httpx 未安装，无法下载图片。请安装: pip install httpx"
            )
        except Exception as e:
            # 网络错误、超时等，记录日志但不阻塞请求
            logger.warning(
                "下载图片失败，保留原始 URL: %s (错误: %s)", url, e,
            )


# =====================================================================
# RoleConsolidationInterceptor
# =====================================================================

class RoleConsolidationInterceptor(BaseInterceptor):
    """
    角色整理拦截器 — 处理消息角色和顺序的兼容性问题。

    ========================================================================
      解决的问题
    ========================================================================

    不同 provider 对消息角色和顺序有不同的约束：

    1. **Anthropic 约束**：
       - 不允许连续出现两个相同角色的消息（必须 user/assistant 交替）。
       - tool_result 必须作为 user 角色消息的 content block 发送，
         不能用独立的 tool 角色消息。
       - 不支持 system / developer 角色消息出现在 messages 列表中
         （必须放在顶层 system 字段）。

    2. **Gemini 约束**：
       - 角色只有 user / model，tool 角色映射为 user。
       - 不允许连续两个 user 角色消息。

    3. **OpenAI 约束**：
       - tool 角色消息必须紧跟在对应 assistant 的 tool_calls 之后。

    ========================================================================
      处理策略
    ========================================================================

    根据 target_provider 的不同，执行不同的整理策略：

    对于 Anthropic / Gemini：
      1. 将 system / developer 角色消息提升到 instructions。
      2. 将 tool 角色消息中的 ToolResult 转移到相邻 user 消息中
         （以 ToolResultBlock 的形式嵌入 content）。
      3. 合并连续的同角色消息（将 content 列表拼接）。

    对于 OpenAI / Ollama：
      - 不做特殊处理（这些 provider 对消息顺序的约束较宽松）。

    ========================================================================
      注意事项
    ========================================================================

    - 本拦截器会修改 request.messages 列表（就地修改）。
    - 被合并的消息的 provider / id 字段会丢失（因为合并后只保留一条消息）。
    - 不会修改 request.vendor 和 request.metadata。
    """

    # 需要执行角色整理的 provider 集合
    _STRICT_ALTERNATION_PROVIDERS: Set[str] = {
        "anthropic",
        "google",
        "bedrock",
    }

    @property
    def name(self) -> str:
        return "role_consolidation"

    async def process(
        self,
        request: UniversalRequest,
        target_provider: str,
    ) -> UniversalRequest:
        """
        根据目标 provider 的约束整理消息结构。

        处理流程：
          1. 提升 system / developer 消息到 instructions（如果目标 provider 需要）
          2. 合并 tool 角色消息到相邻 user 消息（如果目标 provider 需要）
          3. 合并连续同角色消息（如果目标 provider 需要）
        """
        if target_provider not in self._STRICT_ALTERNATION_PROVIDERS:
            logger.debug(
                "目标 provider '%s' 不要求严格的角色交替，跳过整理。",
                target_provider,
            )
            return request

        logger.info(
            "目标 provider '%s' 要求严格角色交替，开始整理消息...",
            target_provider,
        )

        # 步骤 1：将 system / developer 消息提升到 instructions
        request = self._hoist_system_messages(request)

        # 步骤 2：将 tool 角色消息合并到 user 消息
        request = self._merge_tool_into_user(request)

        # 步骤 3：合并连续同角色消息
        request = self._merge_consecutive_same_role(request)

        return request

    def _hoist_system_messages(self, request: UniversalRequest) -> UniversalRequest:
        """
        将 messages 中的 system / developer 角色消息提升到 instructions。

        Anthropic、Gemini、Bedrock 都不支持 system 出现在 messages 列表中，
        需要将其内容转移到顶层 instructions 字段。

        处理后 messages 中不再包含 system / developer 角色的消息。
        """
        hoisted: List[ContentBlock] = []
        remaining: List[Message] = []

        for msg in request.messages:
            if msg.role in ("system", "developer"):
                # 将此消息的 content 块提升到 instructions
                hoisted.extend(msg.content)
                logger.debug("提升 %s 角色消息到 instructions。", msg.role)
            else:
                remaining.append(msg)

        if hoisted:
            # 追加到 instructions 末尾（不覆盖已有的 instructions）
            request.instructions = list(request.instructions) + hoisted
            request.messages = remaining

        return request

    def _merge_tool_into_user(self, request: UniversalRequest) -> UniversalRequest:
        """
        将 tool 角色消息中的 ToolResult 合并到相邻的 user 消息。

        Anthropic 的规则：
          - tool_result 必须出现在 user 角色消息的 content 数组中。
          - 如果连续有多个 tool 消息（并行工具调用的结果），
            它们应该合并进同一个 user 消息。

        转换前：
          [assistant(tool_calls=[...]), tool(result_1), tool(result_2), user("ok")]

        转换后（Anthropic 风格）：
          [assistant(tool_calls=[...]), user([tool_result_1, tool_result_2]), user("ok")]

        如果 tool 消息后面没有 user 消息，则创建一个新的 user 消息来容纳 tool_results。
        """
        new_messages: List[Message] = []
        # 暂存连续的 tool 消息中提取出的 ToolResultBlock
        pending_tool_result_blocks: List[ContentBlock] = []

        for msg in request.messages:
            if msg.role == "tool":
                # 将 tool 消息的内容转换为 ToolResultBlock
                # 优先从 tool_results 顶层列表提取
                for tr in msg.tool_results:
                    pending_tool_result_blocks.append(
                        ToolResultBlock(result=tr)
                    )
                # 也检查 content 中是否有 ToolResultBlock
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        pending_tool_result_blocks.append(block)

                logger.debug(
                    "缓存 tool 消息中的 %d 个 ToolResult。",
                    len(msg.tool_results) + sum(
                        1 for b in msg.content if isinstance(b, ToolResultBlock)
                    ),
                )
            else:
                # 非 tool 消息：先将暂存的 tool_result_blocks 合并
                if pending_tool_result_blocks:
                    if msg.role == "user":
                        # 直接合并到当前 user 消息的 content 开头
                        msg = Message(
                            role="user",
                            content=pending_tool_result_blocks + list(msg.content),
                            id=msg.id,
                            name=msg.name,
                            tool_calls=msg.tool_calls,
                            tool_results=[],  # 已转为 content block
                            provider=msg.provider,
                        )
                        logger.debug(
                            "合并 %d 个 ToolResultBlock 到 user 消息。",
                            len(pending_tool_result_blocks),
                        )
                    else:
                        # 前面有 tool 消息但下一条不是 user → 创建新 user 消息
                        new_messages.append(
                            Message(
                                role="user",
                                content=pending_tool_result_blocks,
                            )
                        )
                        logger.debug(
                            "创建新 user 消息以容纳 %d 个 ToolResultBlock。",
                            len(pending_tool_result_blocks),
                        )
                    pending_tool_result_blocks = []

                new_messages.append(msg)

        # 处理末尾残留的 tool 消息（对话以 tool 消息结尾的情况）
        if pending_tool_result_blocks:
            new_messages.append(
                Message(
                    role="user",
                    content=pending_tool_result_blocks,
                )
            )
            logger.debug(
                "将末尾 %d 个 ToolResultBlock 包装为 user 消息。",
                len(pending_tool_result_blocks),
            )

        request.messages = new_messages
        return request

    def _merge_consecutive_same_role(
        self,
        request: UniversalRequest,
    ) -> UniversalRequest:
        """
        合并连续的同角色消息。

        Anthropic 和 Gemini 不允许连续出现两条相同角色的消息，
        例如：[user("A"), user("B")] 会被 API 拒绝。

        合并策略：
          - 将后一条消息的 content 追加到前一条消息的 content 末尾。
          - tool_calls 和 tool_results 也合并。
          - 后一条消息的 id / name / provider 字段被丢弃
            （只保留第一条消息的元数据）。

        转换前：
          [user("A"), user("B"), assistant("C"), assistant("D")]

        转换后：
          [user(["A", "B"]), assistant(["C", "D"])]
        """
        if not request.messages:
            return request

        merged: List[Message] = [request.messages[0]]

        for msg in request.messages[1:]:
            last = merged[-1]

            if msg.role == last.role:
                # 同角色：合并 content、tool_calls、tool_results
                last.content = list(last.content) + list(msg.content)
                last.tool_calls = list(last.tool_calls) + list(msg.tool_calls)
                last.tool_results = list(last.tool_results) + list(msg.tool_results)
                logger.debug(
                    "合并连续 '%s' 角色消息（content 块: %d → %d）。",
                    msg.role,
                    len(msg.content),
                    len(last.content),
                )
            else:
                # 不同角色：直接追加
                merged.append(msg)

        request.messages = merged
        return request
