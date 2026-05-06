"""
UniversalStreamEvent — 统一流式事件（PRD §23）。

设计原则：
  不直接互转 provider 的 SSE event，而是先归一化为 UniversalStreamEvent，
  再由目标 provider 的适配器重新序列化为其 event 格式。
  无法识别的 event 包装为 ProviderBlock 类型的 delta，附加 STREAM_EVENT_DOWNGRADED warning。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from anyllm.schema.content import ContentBlock, ToolCall
from anyllm.schema.usage import Usage

StreamEventType = Literal[
    "response_started",
    "message_started",
    "content_delta",
    "tool_call_started",
    "tool_call_delta",
    "tool_call_completed",
    "message_completed",
    "response_completed",
    "error",
]
"""
统一流式事件类型枚举：
  response_started   : 响应开始（response id、model 等元数据）
  message_started    : 新消息块开始
  content_delta      : 文本/图片增量（delta 字段携带 ContentBlock）
  tool_call_started  : 工具调用开始（name 已知，arguments 尚未完整）
  tool_call_delta    : 工具调用 arguments 增量
  tool_call_completed: 工具调用完整（arguments 已全部接收）
  message_completed  : 消息块结束
  response_completed : 响应结束（携带 usage 统计）
  error              : 流式传输发生错误
"""


class UniversalStreamEvent(BaseModel):
    """
    统一流式事件，包含增量内容或完成状态。

    适配器在 stream_provider_to_uir() 中将 provider SSE event 转换为此格式；
    在 stream_uir_to_provider() 中将此格式转换回 provider event。
    """

    model_config = ConfigDict(populate_by_name=True)

    type: StreamEventType
    """事件类型。"""

    response_id: str | None = None
    """响应 ID，response_started / response_completed 时填充。"""

    message_id: str | None = None
    """消息 ID，message_started / message_completed 时填充。"""

    index: int | None = None
    """内容块索引，用于多并行内容流的排序。"""

    delta: ContentBlock | None = None
    """
    内容增量，content_delta 事件时填充。
    通常为 TextBlock（text 字段含增量文本片段）。
    无法识别的 provider event 包装为 ProviderBlock。
    """

    tool_call: ToolCall | None = None
    """
    工具调用信息：
      tool_call_started   : id 和 name 已知，arguments 可能为空或部分
      tool_call_delta     : arguments 增量字符串（raw_arguments 字段）
      tool_call_completed : 完整的 ToolCall 对象
    """

    usage: Usage | None = None
    """Token 用量，response_completed 时填充。"""

    raw: Any = None
    """provider 原始 event 数据，完整保留用于调试。"""
