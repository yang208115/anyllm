"""
Anthropic Messages API 适配器（PRD §17）。

========================================================================
  API 格式要点
========================================================================

  请求格式：
    {
        "model": "claude-sonnet-4-5-20241022",
        "max_tokens": 1024,                           ← Anthropic 要求必填
        "system": "You are helpful.",                  ← 顶层 system 字段
        "messages": [
            {"role": "user", "content": [             ← content 永远是 block 数组
                {"type": "text", "text": "描述这张图"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "好的"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo"}}
            ]},
            {"role": "user", "content": [             ← tool_result 必须放在 user 消息里
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny", "is_error": false}
            ]}
        ],
        "tools": [{"name": "get_weather", "description": "...", "input_schema": {...}}]
    }

  响应格式：
    {
        "id": "msg_xxx",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20241022",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "toolu_1", "name": "...", "input": {...}}
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }

========================================================================
  与 UIR 的关键映射差异
========================================================================

  1. system 是顶层字段，不在 messages 中 → UIR instructions
  2. content 永远是 block 数组（不像 OpenAI 可以是字符串）
  3. 图片只支持 base64（不支持 URL）→ 需要 ImageResolutionInterceptor 预处理
  4. tool_use 是 content block（不是消息顶层字段）→ UIR 两种表达方式均支持
  5. tool_result 必须是 user 角色消息的 content block → 需要 RoleConsolidationInterceptor
  6. max_tokens 必填（UIR 的 generation.max_output_tokens）
  7. 不支持 system/developer 角色的消息 → 合并到顶层 system 字段
  8. 不允许连续两条相同角色的消息 → RoleConsolidationInterceptor 处理
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from anyllm.adapters.base import BaseAdapter, ProviderCapabilities
from anyllm.capabilities.matrix import ANTHROPIC_CAPABILITIES
from anyllm.conversion.lowering import blocks_to_plain_text
from anyllm.schema.content import (
    AudioBlock,
    ContentBlock,
    FileBlock,
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
    ToolChoice,
    ToolDef,
)
from anyllm.schema.usage import Usage
from anyllm.schema.warnings import ConversionResult, ConversionWarning

logger = logging.getLogger("anyllm.adapters.anthropic")

# Anthropic 默认 max_tokens（当 UIR 未指定时使用）
_DEFAULT_MAX_TOKENS = 4096


class AnthropicAdapter(BaseAdapter):
    """
    Anthropic Messages API 适配器。

    支持 Claude 3.x / Claude 4.x 系列模型。
    建议搭配 ImageResolutionInterceptor 和 RoleConsolidationInterceptor 使用，
    以确保图片格式和消息角色满足 Anthropic 的约束。
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ANTHROPIC_CAPABILITIES

    # ==================================================================
    # request_to_uir: Anthropic dict → UIR UniversalRequest
    # ==================================================================

    def request_to_uir(
        self,
        raw_request: Dict[str, Any],
    ) -> ConversionResult[UniversalRequest]:
        """
        将 Anthropic Messages API 原始请求转换为 UIR。

        转换要点：
          - 顶层 system 字段 → UIR instructions
          - content block 数组中的 tool_use → ToolCallBlock 或 Message.tool_calls
          - content block 数组中的 tool_result → ToolResultBlock（保留在 user 消息中）
          - image block (base64) → ImageBlock
          - thinking block → ThinkingBlock
        """
        warnings: List[ConversionWarning] = []
        instructions: List[ContentBlock] = []
        messages: List[Message] = []

        # ---- 解析 system 指令 ----
        system = raw_request.get("system")
        if system:
            if isinstance(system, str):
                instructions.append(TextBlock(text=system))
            elif isinstance(system, list):
                # Anthropic 也支持 system 为 block 数组
                for block in system:
                    if block.get("type") == "text":
                        instructions.append(TextBlock(text=block.get("text", "")))

        # ---- 解析 messages ----
        for idx, msg in enumerate(raw_request.get("messages", [])):
            role = msg.get("role", "user")
            path = f"messages[{idx}]"

            content: List[ContentBlock] = []
            tool_calls: List[ToolCall] = []
            tool_results: List[ToolResult] = []

            # Anthropic 的 content 永远是数组
            for j, block in enumerate(msg.get("content", [])):
                block_path = f"{path}.content[{j}]"
                block_type = block.get("type", "")

                if block_type == "text":
                    content.append(TextBlock(text=block.get("text", "")))

                elif block_type == "image":
                    # Anthropic 图片格式：{"type": "image", "source": {"type": "base64", ...}}
                    src = block.get("source", {})
                    content.append(ImageBlock(
                        source=MediaSource(
                            kind=src.get("type", "base64"),
                            value=src.get("data", ""),
                            mime_type=src.get("media_type"),
                        ),
                    ))

                elif block_type == "tool_use":
                    # 工具调用 → 同时放入 content (ToolCallBlock) 和 tool_calls
                    call = ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                        provider={"raw": block},
                    )
                    content.append(ToolCallBlock(call=call))
                    tool_calls.append(call)

                elif block_type == "tool_result":
                    # 工具结果 → 放入 content (ToolResultBlock) 和 tool_results
                    result_content_raw = block.get("content", "")
                    if isinstance(result_content_raw, str):
                        result_content = [TextBlock(text=result_content_raw)]
                    elif isinstance(result_content_raw, list):
                        result_content = []
                        for rc in result_content_raw:
                            if rc.get("type") == "text":
                                result_content.append(TextBlock(text=rc.get("text", "")))
                            elif rc.get("type") == "image":
                                src = rc.get("source", {})
                                result_content.append(ImageBlock(
                                    source=MediaSource(
                                        kind=src.get("type", "base64"),
                                        value=src.get("data", ""),
                                        mime_type=src.get("media_type"),
                                    ),
                                ))
                    else:
                        result_content = [TextBlock(text=str(result_content_raw))]

                    result = ToolResult(
                        call_id=block.get("tool_use_id", ""),
                        content=result_content,
                        is_error=block.get("is_error", False),
                        provider={"raw": block},
                    )
                    content.append(ToolResultBlock(result=result))
                    tool_results.append(result)

                elif block_type == "thinking":
                    content.append(ThinkingBlock(
                        text=block.get("thinking"),
                        encrypted=block.get("encrypted"),
                        signature=block.get("signature"),
                    ))

                else:
                    # 未知 block → ProviderBlock 透传
                    content.append(ProviderBlock(provider="anthropic", value=block))
                    warnings.append(ConversionWarning(
                        code="VENDOR_FIELD_PASSTHROUGH",
                        path=block_path,
                        message=f"未知的 Anthropic content block 类型: '{block_type}'。",
                        severity="info",
                    ))

            messages.append(Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_results=tool_results,
                provider={"raw": msg},
            ))

        # ---- 解析 tools ----
        tools = []
        for i, t in enumerate(raw_request.get("tools") or []):
            tools.append(ToolDef(
                type="function",
                name=t.get("name", ""),
                description=t.get("description"),
                input_schema=t.get("input_schema"),
                provider={k: v for k, v in t.items() if k not in ("name", "description", "input_schema")},
            ))

        # ---- 解析 tool_choice ----
        tool_choice = self._parse_tool_choice(
            raw_request.get("tool_choice"), warnings
        )

        # ---- 解析 generation config ----
        generation = GenerationConfig(
            temperature=raw_request.get("temperature"),
            top_p=raw_request.get("top_p"),
            top_k=raw_request.get("top_k"),
            max_output_tokens=raw_request.get("max_tokens"),
            stop=[raw_request["stop_sequences"]] if isinstance(raw_request.get("stop_sequences"), str)
                else raw_request.get("stop_sequences"),
        )

        uir = UniversalRequest(
            version="uai.v1",
            model=ModelRef(provider="anthropic", name=raw_request.get("model", "")),
            instructions=instructions,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            generation=generation,
            stream=bool(raw_request.get("stream", False)),
            vendor={"anthropic": {
                k: v for k, v in raw_request.items()
                if k not in (
                    "model", "messages", "system", "tools", "tool_choice",
                    "temperature", "top_p", "top_k", "max_tokens",
                    "stop_sequences", "stream",
                )
            }},
        )

        return ConversionResult(value=uir, warnings=warnings)

    # ==================================================================
    # response_to_uir: Anthropic 响应 dict → UIR UniversalResponse
    # ==================================================================

    def response_to_uir(
        self,
        raw_response: Dict[str, Any],
    ) -> ConversionResult[UniversalResponse]:
        """
        将 Anthropic Messages API 原始响应转换为 UIR。

        转换要点：
          - content[].type="text" → TextBlock
          - content[].type="tool_use" → ToolCallBlock + Message.tool_calls
          - content[].type="thinking" → ThinkingBlock
          - stop_reason → normalize_stop_reason("anthropic", ...)
          - usage → UIR Usage（含 cache 信息）
        """
        warnings: List[ConversionWarning] = []

        content: List[ContentBlock] = []
        tool_calls: List[ToolCall] = []

        for block in raw_response.get("content", []):
            block_type = block.get("type", "")

            if block_type == "text":
                content.append(TextBlock(text=block.get("text", "")))

            elif block_type == "tool_use":
                call = ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                )
                content.append(ToolCallBlock(call=call))
                tool_calls.append(call)

            elif block_type == "thinking":
                content.append(ThinkingBlock(
                    text=block.get("thinking"),
                    encrypted=block.get("encrypted"),
                    signature=block.get("signature"),
                ))

            else:
                content.append(ProviderBlock(provider="anthropic", value=block))

        output_msg = Message(
            role=raw_response.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls,
        )

        # ---- 解析 usage ----
        raw_usage = raw_response.get("usage") or {}
        usage = Usage(
            input_tokens=raw_usage.get("input_tokens"),
            output_tokens=raw_usage.get("output_tokens"),
            cached_input_tokens=raw_usage.get("cache_read_input_tokens"),
            provider=raw_usage,
        ) if raw_usage else None

        response = UniversalResponse(
            id=raw_response.get("id"),
            model=raw_response.get("model"),
            output=[output_msg],
            stop_reason=normalize_stop_reason(
                "anthropic", raw_response.get("stop_reason")
            ),
            usage=usage,
            raw=raw_response,
        )

        return ConversionResult(value=response, warnings=warnings)

    # ==================================================================
    # uir_to_request: UIR UniversalRequest → Anthropic dict
    # ==================================================================

    def uir_to_request(
        self,
        request: UniversalRequest,
    ) -> ConversionResult[Dict[str, Any]]:
        """
        将 UIR 转换为 Anthropic Messages API 请求格式。

        转换要点：
          - instructions → 顶层 system 字段（纯文本）
          - system/developer 角色消息 → 合并到顶层 system
          - Message.tool_calls → content 中的 tool_use block
          - Message.tool_results → content 中的 tool_result block
          - ImageBlock → base64 格式（需 ImageResolutionInterceptor 预处理）
          - max_tokens 必填（UIR 未指定时使用默认值 4096）
          - 不允许连续同角色消息（需 RoleConsolidationInterceptor 预处理）
        """
        warnings: List[ConversionWarning] = []

        raw: Dict[str, Any] = {
            "model": request.model.name,
            "messages": [],
            # Anthropic 要求 max_tokens 必填
            "max_tokens": request.generation.max_output_tokens or _DEFAULT_MAX_TOKENS,
        }

        # ---- instructions → 顶层 system ----
        if request.instructions:
            raw["system"] = blocks_to_plain_text(
                request.instructions, warnings, "instructions"
            )

        # ---- 转换每条消息 ----
        for idx, msg in enumerate(request.messages):
            path = f"messages[{idx}]"

            # system / developer 消息 → 合并到顶层 system
            if msg.role in ("system", "developer"):
                extra_system = blocks_to_plain_text(
                    msg.content, warnings, f"{path}.content"
                )
                existing = raw.get("system", "")
                raw["system"] = (existing + "\n" + extra_system).strip()
                warnings.append(ConversionWarning(
                    code="ROLE_DOWNGRADED",
                    path=f"{path}.role",
                    message=f"Anthropic 不支持 messages 中的 '{msg.role}' 角色，"
                            f"已合并到顶层 system 字段。",
                ))
                continue

            # 构建 Anthropic 消息
            anthropic_msg: Dict[str, Any] = {
                "role": self._map_role_to_anthropic(msg.role, warnings, f"{path}.role"),
                "content": [],
            }

            # ---- 转换 content blocks ----
            for j, block in enumerate(msg.content):
                block_path = f"{path}.content[{j}]"
                converted = self._block_to_anthropic(block, warnings, block_path)
                if converted is not None:
                    anthropic_msg["content"].append(converted)

            # ---- 转换 Message.tool_calls → tool_use blocks ----
            # 只转换不在 content 中的 tool_calls（避免重复）
            content_call_ids = {
                b.call.id for b in msg.content if isinstance(b, ToolCallBlock)
            }
            for call in msg.tool_calls:
                if call.id not in content_call_ids:
                    anthropic_msg["content"].append({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    })

            # ---- 转换 Message.tool_results → tool_result blocks ----
            # 只转换不在 content 中的 tool_results
            content_result_ids = {
                b.result.call_id for b in msg.content if isinstance(b, ToolResultBlock)
            }
            for result in msg.tool_results:
                if result.call_id not in content_result_ids:
                    anthropic_msg["content"].append({
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": blocks_to_plain_text(
                            result.content, warnings, f"{path}.tool_results"
                        ),
                        "is_error": result.is_error,
                    })

            # 避免发送空 content 的消息
            if not anthropic_msg["content"]:
                anthropic_msg["content"].append({"type": "text", "text": ""})

            raw["messages"].append(anthropic_msg)

        # ---- tools ----
        if request.tools:
            raw["tools"] = [
                self._tool_def_to_anthropic(t)
                for t in request.tools
                if t.type == "function"
            ]

        # ---- tool_choice ----
        if request.tool_choice is not None:
            raw["tool_choice"] = self._tool_choice_to_anthropic(
                request.tool_choice, warnings
            )

        # ---- generation config（除 max_tokens 外的其他参数）----
        self._apply_generation_config(raw, request.generation)

        # ---- response_format（Anthropic 不原生支持，需降级）----
        if request.response_format is not None:
            if isinstance(request.response_format, (JsonObjectResponseFormat, JsonSchemaResponseFormat)):
                warnings.append(ConversionWarning(
                    code="RESPONSE_FORMAT_DOWNGRADED",
                    path="response_format",
                    message="Anthropic 不原生支持 json_object / json_schema 格式，"
                            "建议在 system prompt 中说明输出格式要求。",
                ))

        # ---- stream ----
        if request.stream:
            raw["stream"] = True

        # ---- state 兼容性检查 ----
        if request.state.previous_response_id or request.state.conversation_id:
            warnings.append(ConversionWarning(
                code="STATE_NOT_SUPPORTED",
                path="state",
                message="Anthropic 不支持有状态会话（previous_response_id / conversation_id）。",
            ))

        # ---- vendor 透传 ----
        vendor_anthropic = request.vendor.get("anthropic", {})
        if vendor_anthropic:
            raw.update(vendor_anthropic)

        return ConversionResult(value=raw, warnings=warnings)

    # ==================================================================
    # uir_to_response: UIR UniversalResponse → Anthropic 响应 dict
    # ==================================================================

    def uir_to_response(
        self,
        response: UniversalResponse,
    ) -> ConversionResult[Dict[str, Any]]:
        """将 UIR 响应转换回 Anthropic Messages API 响应格式。"""
        warnings: List[ConversionWarning] = []

        # 映射 stop_reason 回 Anthropic 格式
        stop_reason_map = {
            "end_turn": "end_turn",
            "max_tokens": "max_tokens",
            "stop_sequence": "stop_sequence",
            "tool_calls": "tool_use",
        }
        stop_reason = stop_reason_map.get(response.stop_reason, "end_turn")

        # 取第一条输出消息
        content_blocks: List[Dict[str, Any]] = []
        if response.output:
            msg = response.output[0]
            for block in msg.content:
                converted = self._block_to_anthropic(block, warnings, "output.content")
                if converted is not None:
                    content_blocks.append(converted)
            # 还原 tool_calls → tool_use blocks
            for call in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                })

        raw: Dict[str, Any] = {
            "id": response.id or "",
            "type": "message",
            "role": "assistant",
            "model": response.model or "",
            "content": content_blocks,
            "stop_reason": stop_reason,
        }

        if response.usage:
            raw["usage"] = {
                "input_tokens": response.usage.input_tokens or 0,
                "output_tokens": response.usage.output_tokens or 0,
            }

        return ConversionResult(value=raw, warnings=warnings)

    # ==================================================================
    # 内部解析方法：Anthropic → UIR
    # ==================================================================

    def _parse_tool_choice(
        self,
        raw: Any,
        warnings: List[ConversionWarning],
    ) -> Optional[ToolChoice]:
        """
        解析 Anthropic tool_choice 字段。

        Anthropic 格式：
          {"type": "auto"}     → AutoToolChoice
          {"type": "none"}     → NoneToolChoice  （Anthropic 实际不支持 none，但保留兼容）
          {"type": "any"}      → RequiredToolChoice
          {"type": "tool", "name": "xxx"} → SpecificToolChoice
        """
        if raw is None:
            return None
        if isinstance(raw, dict):
            choice_type = raw.get("type", "auto")
            if choice_type == "auto":
                return AutoToolChoice()
            if choice_type == "none":
                return NoneToolChoice()
            if choice_type == "any":
                return RequiredToolChoice()
            if choice_type == "tool":
                return SpecificToolChoice(name=raw.get("name", ""))
        return AutoToolChoice()

    # ==================================================================
    # 内部转换方法：UIR → Anthropic
    # ==================================================================

    def _map_role_to_anthropic(
        self,
        role: str,
        warnings: List[ConversionWarning],
        path: str,
    ) -> str:
        """
        将 UIR 角色映射为 Anthropic 角色。

        Anthropic 只支持 user 和 assistant 两个角色：
          user      → user
          assistant → assistant
          tool      → user（tool_result 必须在 user 消息中，由 RoleConsolidationInterceptor 处理）
        """
        if role in ("user", "assistant"):
            return role
        if role == "tool":
            # tool 角色映射为 user（Anthropic 要求 tool_result 在 user 消息中）
            return "user"
        warnings.append(ConversionWarning(
            code="UNSUPPORTED_ROLE",
            path=path,
            message=f"Anthropic 不支持 '{role}' 角色，已降级为 'user'。",
        ))
        return "user"

    def _block_to_anthropic(
        self,
        block: ContentBlock,
        warnings: List[ConversionWarning],
        path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        将单个 UIR ContentBlock 转换为 Anthropic content block dict。

        返回 None 表示跳过此 block（不加入 content 数组）。
        """
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}

        if isinstance(block, ImageBlock):
            if block.source.kind == "base64":
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.source.mime_type or "image/png",
                        "data": block.source.value,
                    },
                }
            # 非 base64 图片 → 降级为文本（应由 ImageResolutionInterceptor 预处理）
            warnings.append(ConversionWarning(
                code="IMAGE_SOURCE_DOWNGRADED",
                path=path,
                message=f"Anthropic 仅支持 base64 图片，当前来源类型为 '{block.source.kind}'。"
                        f"请确保已启用 ImageResolutionInterceptor。",
            ))
            return {"type": "text", "text": f"[Image: {block.source.value}]"}

        if isinstance(block, ToolCallBlock):
            return {
                "type": "tool_use",
                "id": block.call.id,
                "name": block.call.name,
                "input": block.call.arguments,
            }

        if isinstance(block, ToolResultBlock):
            return {
                "type": "tool_result",
                "tool_use_id": block.result.call_id,
                "content": blocks_to_plain_text(
                    block.result.content, warnings, path
                ),
                "is_error": block.result.is_error,
            }

        if isinstance(block, ThinkingBlock):
            result: Dict[str, Any] = {"type": "thinking"}
            if block.text:
                result["thinking"] = block.text
            if block.encrypted:
                result["encrypted"] = block.encrypted
            if block.signature:
                result["signature"] = block.signature
            return result

        if isinstance(block, RefusalBlock):
            # Anthropic 无原生 refusal 概念，转为文本
            return {"type": "text", "text": f"[Refusal: {block.text}]"}

        if isinstance(block, ProviderBlock):
            # 如果是来自 Anthropic 的 ProviderBlock，原样还原
            if block.provider == "anthropic" and isinstance(block.value, dict):
                return block.value
            warnings.append(ConversionWarning(
                code="UNSUPPORTED_CONTENT_BLOCK",
                path=path,
                message=f"来自 '{block.provider}' 的私有块无法转为 Anthropic 格式。",
            ))
            return {"type": "text", "text": "[Unsupported content]"}

        # AudioBlock / FileBlock 等不支持的类型
        warnings.append(ConversionWarning(
            code="UNSUPPORTED_MODALITY",
            path=path,
            message=f"Anthropic 不支持 {type(block).__name__}，已降级为占位文本。",
        ))
        return {"type": "text", "text": f"[{type(block).__name__} omitted]"}

    def _tool_def_to_anthropic(self, tool: ToolDef) -> Dict[str, Any]:
        """将 UIR ToolDef 转换为 Anthropic tools[] 元素。"""
        result: Dict[str, Any] = {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema or {
                "type": "object",
                "properties": {},
            },
        }
        # 透传 Anthropic 特有参数（如 cache_control）
        for key in ("cache_control",):
            if key in tool.provider:
                result[key] = tool.provider[key]
        return result

    def _tool_choice_to_anthropic(
        self,
        choice: ToolChoice,
        warnings: List[ConversionWarning],
    ) -> Dict[str, Any]:
        """
        将 UIR ToolChoice 转换为 Anthropic tool_choice 字段。

        映射：
          AutoToolChoice     → {"type": "auto"}
          NoneToolChoice     → 不设置（Anthropic 通过不传 tools 来禁用）
          RequiredToolChoice → {"type": "any"}
          SpecificToolChoice → {"type": "tool", "name": ...}
        """
        if isinstance(choice, AutoToolChoice):
            return {"type": "auto"}
        if isinstance(choice, NoneToolChoice):
            warnings.append(ConversionWarning(
                code="TOOL_CHOICE_DOWNGRADED",
                path="tool_choice",
                message="Anthropic 不支持 tool_choice='none'，建议通过不传 tools 来禁用工具。",
                severity="info",
            ))
            return {"type": "auto"}
        if isinstance(choice, RequiredToolChoice):
            return {"type": "any"}
        if isinstance(choice, SpecificToolChoice):
            return {"type": "tool", "name": choice.name}
        return {"type": "auto"}

    def _apply_generation_config(
        self,
        raw: Dict[str, Any],
        gen: GenerationConfig,
    ) -> None:
        """将 UIR GenerationConfig 映射到 Anthropic 请求字段。"""
        # max_tokens 已在 uir_to_request 中处理

        if gen.temperature is not None:
            # Anthropic temperature 范围 [0, 1]，UIR 允许 [0, 2]
            raw["temperature"] = min(gen.temperature, 1.0)

        if gen.top_p is not None:
            raw["top_p"] = gen.top_p

        if gen.top_k is not None:
            raw["top_k"] = gen.top_k

        if gen.stop is not None:
            raw["stop_sequences"] = gen.stop

        # Anthropic 不支持的参数 → 静默忽略（不发 warning，属于正常映射差异）
        # seed, presence_penalty, frequency_penalty
