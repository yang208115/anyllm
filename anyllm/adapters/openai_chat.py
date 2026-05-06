"""
OpenAI Chat Completions 适配器（PRD §15）。

========================================================================
  API 格式要点
========================================================================

  请求格式：
    {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": [           ← 多模态时 content 是数组
                {"type": "text", "text": "描述这张图"},
                {"type": "image_url", "image_url": {"url": "https://...", "detail": "high"}}
            ]},
            {"role": "assistant", "content": "...", "tool_calls": [   ← tool_calls 在消息顶层
                {"id": "call_1", "type": "function", "function": {"name": "...", "arguments": "..."}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "..."},   ← tool 结果是独立消息
        ],
        "tools": [{"type": "function", "function": {"name": ..., "parameters": ...}}],
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": false
    }

  响应格式：
    {
        "id": "chatcmpl-xxx",
        "model": "gpt-4o-2024-08-06",
        "choices": [{"index": 0, "message": {...}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }

========================================================================
  与 UIR 的映射差异
========================================================================

  1. system role → UIR instructions（提升到顶层）
  2. tool_calls 在消息顶层（非 content block）→ UIR Message.tool_calls
  3. tool 角色消息 → UIR Message(role="tool", tool_results=[...])
  4. content 可以是 str 或 list → UIR 统一为 List[ContentBlock]
  5. image_url.url 可以是普通 URL 或 data URI → UIR ImageBlock.source
  6. function.arguments 是 JSON 字符串 → UIR ToolCall.arguments 是 dict
"""

from __future__ import annotations

import logging
from typing import Any

from anyllm.adapters.base import BaseAdapter, ProviderCapabilities
from anyllm.capabilities.matrix import OPENAI_CHAT_CAPABILITIES
from anyllm.conversion.lowering import (
    blocks_to_plain_text,
    serialize_tool_arguments,
)
from anyllm.schema.content import (
    ContentBlock,
    ImageBlock,
    MediaSource,
    ProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResult,
    ToolResultBlock,
    parse_tool_arguments,
)
from anyllm.schema.message import Message
from anyllm.schema.request import (
    GenerationConfig,
    ModelRef,
    UniversalRequest,
)
from anyllm.schema.response import UniversalResponse, normalize_stop_reason
from anyllm.schema.tools import (
    AutoToolChoice,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    NoneToolChoice,
    RequiredToolChoice,
    ResponseFormat,
    SpecificToolChoice,
    TextResponseFormat,
    ToolChoice,
    ToolDef,
)
from anyllm.schema.usage import Usage
from anyllm.schema.warnings import ConversionResult, ConversionWarning

logger = logging.getLogger("anyllm.adapters.openai_chat")

# OpenAI Chat 已知的顶层请求字段，用于提取未知字段到 vendor 透传区
_KNOWN_REQUEST_FIELDS = {
    "model", "messages", "tools", "tool_choice", "temperature",
    "top_p", "max_tokens", "stop", "stream", "presence_penalty",
    "frequency_penalty", "seed", "response_format", "n",
    "logprobs", "top_logprobs", "logit_bias", "user",
    "parallel_tool_calls", "store", "metadata",
}


class OpenAIChatAdapter(BaseAdapter):
    """
    OpenAI Chat Completions API 适配器。

    支持 GPT-4o、GPT-4、GPT-3.5-turbo 等模型。
    也可作为 Ollama / Cloudflare Workers AI 等 OpenAI-compatible API 的基类。
    """

    @property
    def provider_name(self) -> str:
        return "openai_chat"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return OPENAI_CHAT_CAPABILITIES

    # ==================================================================
    # request_to_uir: OpenAI Chat dict → UIR UniversalRequest
    # ==================================================================

    def request_to_uir(
        self,
        raw_request: dict[str, Any],
    ) -> ConversionResult[UniversalRequest]:
        """
        将 OpenAI Chat Completions 原始请求转换为 UIR。

        转换要点：
          - role=system 的消息 → 提升到 UIR instructions
          - content: str → [TextBlock(text=...)]
          - content: list → 逐个解析为 TextBlock / ImageBlock / ProviderBlock
          - tool_calls → Message.tool_calls（arguments JSON 字符串解析为 dict）
          - role=tool 消息 → Message(role="tool", tool_results=[...])
        """
        warnings: list[ConversionWarning] = []
        messages: list[Message] = []
        instructions: list[ContentBlock] = []

        # ---- 解析 messages ----
        for idx, msg in enumerate(raw_request.get("messages", [])):
            role = msg.get("role", "user")
            path = f"messages[{idx}]"

            # 解析 content（str 或 list 或 None）
            content = self._parse_content(
                msg.get("content"), warnings, f"{path}.content"
            )

            # system / developer 消息 → 提升到 instructions
            if role in ("system", "developer"):
                instructions.extend(content)
                continue

            # 解析 tool_calls（assistant 消息可能携带）
            tool_calls: list[ToolCall] = []
            for tc_idx, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                # OpenAI 的 arguments 是 JSON 字符串，需要解析为 dict
                args, raw_args, warn_code = parse_tool_arguments(
                    fn.get("arguments", "{}")
                )
                if warn_code:
                    warnings.append(ConversionWarning(
                        code=warn_code,
                        path=f"{path}.tool_calls[{tc_idx}].function.arguments",
                        message="工具调用参数不是合法的 JSON 字符串。",
                    ))
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                    raw_arguments=raw_args,
                    provider={"raw": tc},
                ))

            # 解析 tool 角色消息的 tool_call_id → ToolResult
            tool_results: list[ToolResult] = []
            if role == "tool" and "tool_call_id" in msg:
                tool_results.append(ToolResult(
                    call_id=msg["tool_call_id"],
                    content=[TextBlock(text=msg.get("content", "") or "")],
                    name=msg.get("name"),
                ))
                # tool 角色的 content 已作为 ToolResult.content，清空避免重复
                content = []

            # 检查 refusal 字段（OpenAI 安全拒绝）
            if msg.get("refusal"):
                content.append(RefusalBlock(text=msg["refusal"]))

            messages.append(Message(
                role=role,
                content=content,
                name=msg.get("name"),
                tool_calls=tool_calls,
                tool_results=tool_results,
                provider={"raw": msg},
            ))

        # ---- 解析 tools ----
        tools = [
            self._parse_tool_def(t, warnings, f"tools[{i}]")
            for i, t in enumerate(raw_request.get("tools") or [])
        ]

        # ---- 解析 tool_choice ----
        tool_choice = self._parse_tool_choice(
            raw_request.get("tool_choice"), warnings
        )

        # ---- 解析 response_format ----
        response_format = self._parse_response_format(
            raw_request.get("response_format"), warnings
        )

        # ---- 解析 generation config ----
        generation = GenerationConfig(
            temperature=raw_request.get("temperature"),
            top_p=raw_request.get("top_p"),
            max_output_tokens=raw_request.get("max_tokens"),
            stop=raw_request.get("stop"),
            seed=raw_request.get("seed"),
            presence_penalty=raw_request.get("presence_penalty"),
            frequency_penalty=raw_request.get("frequency_penalty"),
        )

        # ---- 提取未知字段到 vendor 透传区 ----
        vendor_fields = {
            k: v for k, v in raw_request.items()
            if k not in _KNOWN_REQUEST_FIELDS
        }

        uir = UniversalRequest(
            version="uai.v1",
            model=ModelRef(provider="openai", name=raw_request.get("model", "")),
            instructions=instructions,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            generation=generation,
            stream=bool(raw_request.get("stream", False)),
            vendor={"openai_chat": vendor_fields} if vendor_fields else {},
        )

        return ConversionResult(value=uir, warnings=warnings)

    # ==================================================================
    # response_to_uir: OpenAI Chat 响应 dict → UIR UniversalResponse
    # ==================================================================

    def response_to_uir(
        self,
        raw_response: dict[str, Any],
    ) -> ConversionResult[UniversalResponse]:
        """
        将 OpenAI Chat Completions 原始响应转换为 UIR。

        转换要点：
          - choices[0].message → UIR Message
          - finish_reason → normalize_stop_reason("openai", ...)
          - usage → UIR Usage
          - tool_calls → Message.tool_calls
        """
        warnings: list[ConversionWarning] = []
        output: list[Message] = []

        for choice in raw_response.get("choices", []):
            msg = choice.get("message", {})

            # 解析 content
            content = self._parse_content(
                msg.get("content"), warnings, "choices.message.content"
            )

            # 解析 refusal
            if msg.get("refusal"):
                content.append(RefusalBlock(text=msg["refusal"]))

            # 解析 tool_calls
            tool_calls: list[ToolCall] = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                args, raw_args, warn_code = parse_tool_arguments(
                    fn.get("arguments", "{}")
                )
                if warn_code:
                    warnings.append(ConversionWarning(
                        code=warn_code,
                        path="choices.message.tool_calls",
                        message="响应中的工具调用参数不是合法 JSON。",
                    ))
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                    raw_arguments=raw_args,
                ))

            output.append(Message(
                role=msg.get("role", "assistant"),
                content=content,
                tool_calls=tool_calls,
            ))

        # ---- 解析 usage ----
        raw_usage = raw_response.get("usage") or {}
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens"),
            output_tokens=raw_usage.get("completion_tokens"),
            total_tokens=raw_usage.get("total_tokens"),
            provider=raw_usage,
        ) if raw_usage else None

        # ---- 解析 stop reason ----
        finish_reason = None
        if raw_response.get("choices"):
            finish_reason = raw_response["choices"][0].get("finish_reason")

        response = UniversalResponse(
            id=raw_response.get("id"),
            model=raw_response.get("model"),
            output=output,
            stop_reason=normalize_stop_reason("openai", finish_reason),
            usage=usage,
            raw=raw_response,
        )

        return ConversionResult(value=response, warnings=warnings)

    # ==================================================================
    # uir_to_request: UIR UniversalRequest → OpenAI Chat dict
    # ==================================================================

    def uir_to_request(
        self,
        request: UniversalRequest,
    ) -> ConversionResult[dict[str, Any]]:
        """
        将 UIR 转换为 OpenAI Chat Completions 请求格式。

        转换要点：
          - instructions → 首条 role=system 消息
          - Message.tool_calls → 消息顶层 tool_calls 字段（arguments 序列化为 JSON 字符串）
          - Message(role="tool") → 独立消息 + tool_call_id
          - ToolCallBlock in content → 提取到消息顶层 tool_calls
          - ImageBlock → image_url 格式（支持 url 和 data URI base64）
        """
        warnings: list[ConversionWarning] = []
        messages: list[dict[str, Any]] = []

        # ---- instructions → system 消息 ----
        if request.instructions:
            messages.append({
                "role": "system",
                "content": blocks_to_plain_text(
                    request.instructions, warnings, "instructions"
                ),
            })

        # ---- 转换每条消息 ----
        for idx, msg in enumerate(request.messages):
            path = f"messages[{idx}]"
            role = self._map_role_to_openai(msg.role, warnings, f"{path}.role")

            # 转换 content blocks → OpenAI content 格式
            openai_content = self._blocks_to_openai_content(
                msg.content, warnings, f"{path}.content"
            )

            out: dict[str, Any] = {
                "role": role,
                "content": openai_content,
            }

            # 保留 name 字段（多 agent 场景）
            if msg.name:
                out["name"] = msg.name

            # ---- 处理 tool_calls ----
            # 合并 Message.tool_calls 和 content 中的 ToolCallBlock
            all_tool_calls = list(msg.tool_calls)
            for block in msg.content:
                if isinstance(block, ToolCallBlock):
                    all_tool_calls.append(block.call)

            if all_tool_calls:
                out["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": serialize_tool_arguments(call.arguments),
                        },
                    }
                    for call in all_tool_calls
                ]

            # ---- 处理 tool 角色消息 ----
            # OpenAI 的 tool 消息需要 tool_call_id 字段，content 必须是字符串
            if role == "tool" and msg.tool_results:
                result = msg.tool_results[0]
                out["tool_call_id"] = result.call_id
                out["content"] = blocks_to_plain_text(
                    result.content, warnings, f"{path}.tool_results[0].content"
                )
            # 也处理 content 中的 ToolResultBlock（Anthropic 风格转来的情况）
            elif role == "tool":
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        out["tool_call_id"] = block.result.call_id
                        out["content"] = blocks_to_plain_text(
                            block.result.content, warnings, f"{path}.content"
                        )
                        break

            messages.append(out)

        # ---- 组装最终请求 ----
        raw: dict[str, Any] = {
            "model": request.model.name,
            "messages": messages,
        }

        # ---- tools ----
        if request.tools:
            raw["tools"] = [
                self._tool_def_to_openai(t) for t in request.tools
            ]

        # ---- tool_choice ----
        if request.tool_choice is not None:
            raw["tool_choice"] = self._tool_choice_to_openai(
                request.tool_choice, warnings
            )

        # ---- response_format ----
        if request.response_format is not None:
            raw["response_format"] = self._response_format_to_openai(
                request.response_format, warnings
            )

        # ---- generation config ----
        self._apply_generation_config(raw, request.generation)

        # ---- stream ----
        raw["stream"] = request.stream

        # ---- state 兼容性检查 ----
        if request.state.previous_response_id or request.state.conversation_id:
            warnings.append(ConversionWarning(
                code="STATE_NOT_SUPPORTED",
                path="state",
                message="OpenAI Chat Completions 不支持 previous_response_id 或有状态会话。",
            ))

        # ---- vendor 透传 ----
        vendor_openai = request.vendor.get("openai_chat", {})
        if vendor_openai:
            raw.update(vendor_openai)

        return ConversionResult(value=raw, warnings=warnings)

    # ==================================================================
    # uir_to_response: UIR UniversalResponse → OpenAI Chat 响应 dict
    # ==================================================================

    def uir_to_response(
        self,
        response: UniversalResponse,
    ) -> ConversionResult[dict[str, Any]]:
        """将 UIR 响应转换回 OpenAI Chat Completions 响应格式。"""
        warnings: list[ConversionWarning] = []

        # 映射 stop_reason 回 OpenAI 格式
        stop_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_calls": "tool_calls",
            "content_filter": "content_filter",
        }
        finish_reason = stop_reason_map.get(response.stop_reason, "stop")

        choices = []
        for i, msg in enumerate(response.output):
            choice_msg: dict[str, Any] = {
                "role": msg.role,
                "content": self._blocks_to_openai_content(
                    msg.content, warnings, f"output[{i}].content"
                ),
            }

            # 还原 tool_calls
            all_calls = list(msg.tool_calls)
            for block in msg.content:
                if isinstance(block, ToolCallBlock):
                    all_calls.append(block.call)

            if all_calls:
                choice_msg["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": serialize_tool_arguments(call.arguments),
                        },
                    }
                    for call in all_calls
                ]

            # content 如果只有一个 TextBlock，简化为字符串
            content = choice_msg["content"]
            if (
                isinstance(content, list)
                and len(content) == 1
                and isinstance(content[0], dict)
                and content[0].get("type") == "text"
            ):
                choice_msg["content"] = content[0]["text"]

            choices.append({
                "index": i,
                "message": choice_msg,
                "finish_reason": finish_reason,
            })

        raw: dict[str, Any] = {
            "id": response.id or "",
            "object": "chat.completion",
            "model": response.model or "",
            "choices": choices,
        }

        if response.usage:
            raw["usage"] = {
                "prompt_tokens": response.usage.input_tokens or 0,
                "completion_tokens": response.usage.output_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return ConversionResult(value=raw, warnings=warnings)

    # ==================================================================
    # 内部解析方法：OpenAI → UIR
    # ==================================================================

    def _parse_content(
        self,
        content: Any,
        warnings: list[ConversionWarning],
        path: str,
    ) -> list[ContentBlock]:
        """
        解析 OpenAI 的 content 字段。

        OpenAI content 有三种形态：
          - None        → 空列表（assistant 只含 tool_calls 时 content 为 null）
          - str         → [TextBlock(text=...)]
          - list[dict]  → 逐个解析：text / image_url / input_image / 其他
        """
        if content is None:
            return []

        if isinstance(content, str):
            return [TextBlock(text=content)] if content else []

        if isinstance(content, list):
            blocks: list[ContentBlock] = []
            for i, part in enumerate(content):
                part_type = part.get("type", "")
                part_path = f"{path}[{i}]"

                if part_type == "text":
                    blocks.append(TextBlock(text=part.get("text", "")))

                elif part_type in ("image_url", "input_image"):
                    # image_url 的结构：{"type": "image_url", "image_url": {"url": ..., "detail": ...}}
                    image_data = part.get("image_url", {})
                    if isinstance(image_data, str):
                        # 简化形式：image_url 直接是 URL 字符串
                        url = image_data
                        detail = None
                    else:
                        url = image_data.get("url", "")
                        detail = image_data.get("detail")

                    # 判断是 data URI (base64) 还是普通 URL
                    if url.startswith("data:"):
                        # data:image/png;base64,iVBOR...
                        # 解析 MIME 类型和 base64 数据
                        header, _, b64_data = url.partition(",")
                        mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                        source = MediaSource(kind="base64", value=b64_data, mime_type=mime_type)
                    else:
                        source = MediaSource(kind="url", value=url)

                    blocks.append(ImageBlock(source=source, detail=detail))

                else:
                    # 未知类型 → ProviderBlock 透传
                    blocks.append(ProviderBlock(provider="openai", value=part))
                    warnings.append(ConversionWarning(
                        code="VENDOR_FIELD_PASSTHROUGH",
                        path=part_path,
                        message=f"未知的 OpenAI content part 类型: '{part_type}'，已透传保留。",
                        severity="info",
                    ))

            return blocks

        # 其他类型 → 强制转为字符串
        warnings.append(ConversionWarning(
            code="UNSUPPORTED_CONTENT",
            path=path,
            message=f"不支持的 OpenAI content 格式: {type(content).__name__}，已强制转为文本。",
        ))
        return [TextBlock(text=str(content))]

    def _parse_tool_def(
        self,
        tool: dict[str, Any],
        warnings: list[ConversionWarning],
        path: str,
    ) -> ToolDef:
        """解析 OpenAI tools[] 元素为 UIR ToolDef。"""
        fn = tool.get("function", {})
        return ToolDef(
            type="function",
            name=fn.get("name", ""),
            description=fn.get("description"),
            input_schema=fn.get("parameters"),
            provider={"strict": fn.get("strict"), "raw": tool},
        )

    def _parse_tool_choice(
        self,
        raw: Any,
        warnings: list[ConversionWarning],
    ) -> ToolChoice | None:
        """
        解析 OpenAI tool_choice 字段。

        OpenAI 格式：
          "auto"     → AutoToolChoice
          "none"     → NoneToolChoice
          "required" → RequiredToolChoice
          {"type": "function", "function": {"name": "xxx"}} → SpecificToolChoice
        """
        if raw is None:
            return None
        if raw == "auto":
            return AutoToolChoice()
        if raw == "none":
            return NoneToolChoice()
        if raw == "required":
            return RequiredToolChoice()
        if isinstance(raw, dict):
            fn = raw.get("function", {})
            name = fn.get("name", "")
            if name:
                return SpecificToolChoice(name=name)
        return AutoToolChoice()

    def _parse_response_format(
        self,
        raw: Any,
        warnings: list[ConversionWarning],
    ) -> ResponseFormat | None:
        """
        解析 OpenAI response_format 字段。

        OpenAI 格式：
          {"type": "text"} → TextResponseFormat
          {"type": "json_object"} → JsonObjectResponseFormat
          {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}
            → JsonSchemaResponseFormat
        """
        if raw is None:
            return None
        fmt_type = raw.get("type", "text")
        if fmt_type == "text":
            return TextResponseFormat()
        if fmt_type == "json_object":
            return JsonObjectResponseFormat()
        if fmt_type == "json_schema":
            js = raw.get("json_schema", {})
            return JsonSchemaResponseFormat(
                name=js.get("name", ""),
                schema=js.get("schema", {}),
                strict=js.get("strict", False),
            )
        return None

    # ==================================================================
    # 内部转换方法：UIR → OpenAI
    # ==================================================================

    def _map_role_to_openai(
        self,
        role: str,
        warnings: list[ConversionWarning],
        path: str,
    ) -> str:
        """
        将 UIR 角色映射为 OpenAI 角色。

        映射规则：
          user      → user
          assistant → assistant
          tool      → tool
          system    → system（通常不会走到这里，因为已提升到 instructions）
          developer → developer（OpenAI Responses API 支持）
        """
        valid_roles = {"user", "assistant", "tool", "system", "developer"}
        if role in valid_roles:
            return role
        warnings.append(ConversionWarning(
            code="UNSUPPORTED_ROLE",
            path=path,
            message=f"未知的角色 '{role}'，已降级为 'user'。",
        ))
        return "user"

    def _blocks_to_openai_content(
        self,
        blocks: list[ContentBlock],
        warnings: list[ConversionWarning],
        path: str,
    ) -> Any:
        """
        将 UIR ContentBlock 列表转换为 OpenAI content 格式。

        规则：
          - 如果只有一个 TextBlock → 返回纯字符串（OpenAI 偏好简单格式）
          - 如果有多个 block 或包含图片 → 返回 content parts 数组
          - ToolCallBlock → 跳过（由消息顶层 tool_calls 字段处理）
          - ToolResultBlock → 跳过（由 tool 角色消息处理）
        """
        if not blocks:
            return None

        # 过滤掉 ToolCallBlock 和 ToolResultBlock（它们由消息顶层字段处理）
        displayable = [
            b for b in blocks
            if not isinstance(b, (ToolCallBlock, ToolResultBlock))
        ]

        if not displayable:
            return None

        # 如果只有一个 TextBlock，返回纯字符串
        if len(displayable) == 1 and isinstance(displayable[0], TextBlock):
            return displayable[0].text

        # 多个 block → 返回 content parts 数组
        parts: list[dict[str, Any]] = []
        for i, block in enumerate(displayable):
            block_path = f"{path}[{i}]"

            if isinstance(block, TextBlock):
                parts.append({"type": "text", "text": block.text})

            elif isinstance(block, ImageBlock):
                parts.append(self._image_to_openai(block))

            elif isinstance(block, RefusalBlock):
                # OpenAI 的 refusal 是消息顶层字段，但这里放在 content 里作为文本
                parts.append({"type": "text", "text": f"[Refusal: {block.text}]"})

            elif isinstance(block, ThinkingBlock):
                # OpenAI 不原生支持 thinking block，降级为文本注释
                if block.text:
                    parts.append({"type": "text", "text": f"[Thinking: {block.text}]"})
                warnings.append(ConversionWarning(
                    code="UNSUPPORTED_CONTENT_BLOCK",
                    path=block_path,
                    message="OpenAI Chat 不支持 thinking block，已降级为文本。",
                    severity="info",
                ))

            elif isinstance(block, ProviderBlock):
                # 如果是来自 OpenAI 的 ProviderBlock，原样还原
                if block.provider == "openai" and isinstance(block.value, dict):
                    parts.append(block.value)
                else:
                    parts.append({"type": "text", "text": "[Unsupported content]"})
                    warnings.append(ConversionWarning(
                        code="UNSUPPORTED_CONTENT_BLOCK",
                        path=block_path,
                        message=f"来自 '{block.provider}' 的私有块无法转为 OpenAI 格式。",
                    ))

            else:
                # AudioBlock / FileBlock 等 → 降级
                parts.append({"type": "text", "text": "[Unsupported content]"})
                warnings.append(ConversionWarning(
                    code="UNSUPPORTED_MODALITY",
                    path=block_path,
                    message=f"OpenAI Chat 不支持 {type(block).__name__}，已降级为占位文本。",
                ))

        return parts

    def _image_to_openai(self, block: ImageBlock) -> dict[str, Any]:
        """
        将 ImageBlock 转换为 OpenAI image_url 格式。

        支持：
          url    → {"type": "image_url", "image_url": {"url": "https://...", "detail": ...}}
          base64 → {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>", "detail": ...}}
        """
        if block.source.kind == "url":
            url = block.source.value
        elif block.source.kind == "base64":
            # 将 base64 包装为 data URI
            mime = block.source.mime_type or "image/png"
            url = f"data:{mime};base64,{block.source.value}"
        else:
            # file_id / bytes → 降级为占位文本
            url = f"[Image: {block.source.kind}]"

        result: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": url},
        }
        if block.detail:
            result["image_url"]["detail"] = block.detail
        return result

    def _tool_def_to_openai(self, tool: ToolDef) -> dict[str, Any]:
        """将 UIR ToolDef 转换为 OpenAI tools[] 元素。"""
        fn: dict[str, Any] = {
            "name": tool.name,
        }
        if tool.description:
            fn["description"] = tool.description
        if tool.input_schema:
            fn["parameters"] = tool.input_schema
        # 透传 strict 参数
        if tool.provider.get("strict"):
            fn["strict"] = tool.provider["strict"]

        return {"type": "function", "function": fn}

    def _tool_choice_to_openai(
        self,
        choice: ToolChoice,
        warnings: list[ConversionWarning],
    ) -> Any:
        """
        将 UIR ToolChoice 转换为 OpenAI tool_choice 字段。

        映射：
          AutoToolChoice     → "auto"
          NoneToolChoice     → "none"
          RequiredToolChoice → "required"
          SpecificToolChoice → {"type": "function", "function": {"name": ...}}
        """
        if isinstance(choice, AutoToolChoice):
            return "auto"
        if isinstance(choice, NoneToolChoice):
            return "none"
        if isinstance(choice, RequiredToolChoice):
            return "required"
        if isinstance(choice, SpecificToolChoice):
            return {
                "type": "function",
                "function": {"name": choice.name},
            }
        return "auto"

    def _response_format_to_openai(
        self,
        fmt: ResponseFormat,
        warnings: list[ConversionWarning],
    ) -> dict[str, Any]:
        """将 UIR ResponseFormat 转换为 OpenAI response_format 字段。"""
        if isinstance(fmt, TextResponseFormat):
            return {"type": "text"}
        if isinstance(fmt, JsonObjectResponseFormat):
            return {"type": "json_object"}
        if isinstance(fmt, JsonSchemaResponseFormat):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.name,
                    "schema": fmt.json_schema,
                    "strict": fmt.strict,
                },
            }
        return {"type": "text"}

    def _apply_generation_config(
        self,
        raw: dict[str, Any],
        gen: GenerationConfig,
    ) -> None:
        """将 UIR GenerationConfig 映射到 OpenAI 请求的顶层字段。"""
        if gen.temperature is not None:
            raw["temperature"] = gen.temperature
        if gen.top_p is not None:
            raw["top_p"] = gen.top_p
        if gen.max_output_tokens is not None:
            raw["max_tokens"] = gen.max_output_tokens
        if gen.stop is not None:
            raw["stop"] = gen.stop
        if gen.seed is not None:
            raw["seed"] = gen.seed
        if gen.presence_penalty is not None:
            raw["presence_penalty"] = gen.presence_penalty
        if gen.frequency_penalty is not None:
            raw["frequency_penalty"] = gen.frequency_penalty
