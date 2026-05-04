"""
降级工具函数 (Lowering Utilities) — PRD §14。

当目标 provider 不支持某些高级内容块（ImageBlock、FileBlock 等）时，
需要将它们「降级」为纯文本占位符，同时发出对应的 ConversionWarning。

这些函数被所有适配器共享，是 uir_to_request / uir_to_response 的基础设施。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from anyllm.schema.content import (
    AudioBlock,
    ContentBlock,
    FileBlock,
    ImageBlock,
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
from anyllm.schema.warnings import ConversionWarning


def blocks_to_plain_text(
    blocks: List[ContentBlock],
    warnings: List[ConversionWarning],
    path: str,
) -> str:
    """
    将 ContentBlock 列表降级为纯文本字符串（PRD §14.1）。

    使用场景：
      - 某些 provider（如 Ollama OpenAI-compat）的 content 只接受字符串
      - System prompt 需要转为纯文本（如 OpenAI Chat 的 system 消息）
      - ToolResult 内容需要序列化为字符串（如 OpenAI Chat 的 tool 消息）

    降级规则：
      TextBlock      → 直接取 text
      ImageBlock     → "[Image omitted]" + UNSUPPORTED_MODALITY warning
      AudioBlock     → "[Audio omitted]" + UNSUPPORTED_MODALITY warning
      FileBlock      → "[File omitted: <filename>]" + FILE_REFERENCE_NOT_SUPPORTED warning
      ThinkingBlock  → "[Thinking omitted]"（思考过程通常不需要传给下一个 provider）
      RefusalBlock   → 直接取 text（拒绝理由应保留）
      ToolCallBlock  → "[Tool call omitted: <name>]" + TOOL_CALL_BLOCK_DOWNGRADED warning
      ToolResultBlock→ "[Tool result omitted]" + TOOL_RESULT_BLOCK_DOWNGRADED warning
      ProviderBlock  → "[Unsupported content omitted]" + UNSUPPORTED_CONTENT_BLOCK warning

    Args:
        blocks   : 待降级的 ContentBlock 列表。
        warnings : 警告收集器（就地追加），调用方负责传入并最终传给 ConversionResult。
        path     : 当前字段路径（如 "messages[2].content"），用于 warning 定位。

    Returns:
        降级后的纯文本字符串，多个 block 之间用换行符连接。
    """
    parts: List[str] = []

    for i, block in enumerate(blocks):
        block_path = f"{path}[{i}]"

        if isinstance(block, TextBlock):
            # 文本块：直接保留
            parts.append(block.text)

        elif isinstance(block, ImageBlock):
            # 图片块：降级为占位符，附加警告
            parts.append("[Image omitted]")
            warnings.append(
                ConversionWarning(
                    code="UNSUPPORTED_MODALITY",
                    path=block_path,
                    message="目标 provider 不支持图片内容块，已降级为占位文本。",
                )
            )

        elif isinstance(block, AudioBlock):
            # 音频块：降级为占位符
            parts.append("[Audio omitted]")
            warnings.append(
                ConversionWarning(
                    code="UNSUPPORTED_MODALITY",
                    path=block_path,
                    message="目标 provider 不支持音频内容块，已降级为占位文本。",
                )
            )

        elif isinstance(block, FileBlock):
            # 文件块：降级，保留文件名信息
            fname = block.filename or block.source.filename or "unknown"
            parts.append(f"[File omitted: {fname}]")
            warnings.append(
                ConversionWarning(
                    code="FILE_REFERENCE_NOT_SUPPORTED",
                    path=block_path,
                    message=f"目标 provider 不支持文件引用块（{fname}），已降级为占位文本。",
                )
            )

        elif isinstance(block, ThinkingBlock):
            # 思考块：通常不传递给下一个 provider，静默降级
            if block.text:
                parts.append(f"[Thinking: {block.text[:100]}...]")

        elif isinstance(block, RefusalBlock):
            # 拒绝块：保留拒绝理由文本
            parts.append(block.text)

        elif isinstance(block, ToolCallBlock):
            # 工具调用块：降级为文本占位符
            parts.append(f"[Tool call omitted: {block.call.name}]")
            warnings.append(
                ConversionWarning(
                    code="TOOL_CALL_BLOCK_DOWNGRADED",
                    path=block_path,
                    message=f"工具调用 '{block.call.name}' 无法表达为目标格式，已降级为占位文本。",
                )
            )

        elif isinstance(block, ToolResultBlock):
            # 工具结果块：降级为文本占位符
            parts.append("[Tool result omitted]")
            warnings.append(
                ConversionWarning(
                    code="TOOL_RESULT_BLOCK_DOWNGRADED",
                    path=block_path,
                    message="工具结果块无法表达为目标格式，已降级为占位文本。",
                )
            )

        elif isinstance(block, ProviderBlock):
            # 厂商私有块：降级
            parts.append("[Unsupported content omitted]")
            warnings.append(
                ConversionWarning(
                    code="UNSUPPORTED_CONTENT_BLOCK",
                    path=block_path,
                    message=f"不支持的 provider 私有块（来源: {block.provider}），已降级为占位文本。",
                )
            )

        else:
            # 未知类型：兜底处理
            parts.append("[Unsupported content omitted]")
            warnings.append(
                ConversionWarning(
                    code="UNSUPPORTED_CONTENT_BLOCK",
                    path=block_path,
                    message=f"未知的内容块类型: {getattr(block, 'type', 'unknown')}",
                )
            )

    return "\n".join(parts)


def tool_result_content_to_text(
    result: ToolResult,
    warnings: List[ConversionWarning],
    path: str,
) -> str:
    """
    将 ToolResult.content 降级为纯文本。

    OpenAI Chat 的 tool 角色消息 content 必须是字符串，
    因此需要将 ToolResult 中可能包含的富文本（ImageBlock 等）降级。

    Args:
        result   : ToolResult 对象。
        warnings : 警告收集器。
        path     : 字段路径，如 "messages[3].tool_results[0].content"。

    Returns:
        纯文本字符串。
    """
    return blocks_to_plain_text(result.content, warnings, path)


def extract_text_from_blocks(blocks: List[ContentBlock]) -> str:
    """
    从 ContentBlock 列表中提取所有 TextBlock 的文本，忽略其他类型。

    这是一个「无损提取」版本，不产生 warning，
    适用于只需要文本内容且不关心其他模态的场景（如日志记录）。

    Args:
        blocks: ContentBlock 列表。

    Returns:
        所有 TextBlock.text 拼接的结果。
    """
    return "\n".join(
        block.text
        for block in blocks
        if isinstance(block, TextBlock)
    )


def serialize_tool_arguments(arguments: Any) -> str:
    """
    将工具调用参数序列化为 JSON 字符串。

    OpenAI Chat / Responses 要求 tool_calls[].function.arguments 为 JSON 字符串，
    而 UIR 内部存储为 dict。此函数负责转换。

    Args:
        arguments: 已解析的参数 dict（或其他类型）。

    Returns:
        JSON 字符串，中文不转义（ensure_ascii=False）。
    """
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)
