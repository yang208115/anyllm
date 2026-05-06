"""
UniversalResponse — 统一响应体（PRD §26）。

将 provider 返回的响应归一化为统一格式，便于上层应用无感切换 provider。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anyllm.schema.message import Message
from anyllm.schema.request import ConversationState, StopReason
from anyllm.schema.usage import Usage

# Stop reason 映射表（PRD §24）
PROVIDER_STOP_REASON_MAP: dict[str, dict[str, StopReason]] = {
    "openai": {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
    },
    "anthropic": {
        "end_turn": "end_turn",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "tool_use": "tool_calls",
    },
    "gemini": {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    },
    "bedrock": {
        "end_turn": "end_turn",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "tool_use": "tool_calls",
    },
}


def normalize_stop_reason(provider: str, raw_reason: str | None) -> StopReason:
    """
    将 provider 原始 stop reason 映射为统一的 StopReason 枚举值。

    Args:
        provider: provider 标识符，例如 "openai"、"anthropic"。
        raw_reason: provider 返回的原始停止原因字符串。

    Returns:
        对应的 StopReason，无法映射时返回 "unknown"。
    """
    if raw_reason is None:
        return "unknown"
    mapping = PROVIDER_STOP_REASON_MAP.get(provider, {})
    return mapping.get(raw_reason, "unknown")


class UniversalResponse(BaseModel):
    """
    AnyLLM 统一响应体。

    output 是一个 Message 列表，通常只有一条 assistant 消息，
    但部分 provider（如 OpenAI Responses API）可能返回多条输出消息。
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    """响应 ID，由 provider 生成。"""

    model: str | None = None
    """实际使用的模型名称（provider 返回，可能与请求的 model 不同）。"""

    output: list[Message] = Field(default_factory=list)
    """模型输出消息列表，通常为单条 assistant 消息。"""

    stop_reason: StopReason = "unknown"
    """统一停止原因，由 normalize_stop_reason() 映射。"""

    usage: Usage | None = None
    """Token 用量统计。"""

    state: ConversationState = Field(default_factory=ConversationState)
    """有状态会话信息，OpenAI Responses / Assistants 会填充此字段。"""

    raw: Any = None
    """provider 原始响应，完整保留用于调试。"""

    vendor: dict[str, Any] = Field(default_factory=dict)
    """厂商特定响应字段透传区，例如 Anthropic 的 cache 统计。"""
