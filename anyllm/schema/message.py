"""
Message — 统一消息结构（PRD §6）。

设计要点：
  一条 Message = role + content[] + tool_calls[] + tool_results[]

  tool_calls 和 tool_results 与 content 并列，而不是只放在 content 里，
  原因是不同 provider 表达工具调用的位置不同：
    OpenAI Chat : assistant.message.tool_calls（顶层） / tool role message
    Anthropic   : assistant content 里的 tool_use block / user content 里的 tool_result block
    Gemini      : model part 里的 functionCall / user part 里的 functionResponse
    Bedrock     : content 里的 toolUse / toolResult

  UIR 同时支持两种表达，适配器在转换时可以互相派生。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from anyllm.schema.content import ContentBlock, ToolCall, ToolResult

Role = Literal["system", "developer", "user", "assistant", "tool"]
"""
统一角色枚举（PRD §5.1）：
  system    : 系统指令（大多数 provider 通过 instructions/system 字段表达，不进入 messages）
  developer : OpenAI Responses API 专用，优先级高于 user 消息，行为类似 system
  user      : 人类用户输入
  assistant : 模型输出
  tool      : 工具执行结果消息

注意：UIR 内部保留全部 5 种角色，适配器负责在输出时将其映射到目标 provider 的角色体系。
例如 Anthropic 不支持 system role 消息，适配器会将其合并进顶层 system 字段并发出 warning。
"""


class Message(BaseModel):
    """
    统一消息格式，支持多模态内容块与工具调用的混合表达。

    content 与 tool_calls / tool_results 的关系：
      - 一条 assistant 消息可以同时含有 TextBlock（回答文本）和 tool_calls（工具调用请求）。
      - 一条 tool 消息通常 content 为空，tool_results 携带执行结果。
      - 一条 user 消息可以含有 TextBlock、ImageBlock，也可以含有 ToolResultBlock（Anthropic 风格）。

    适配器在 uir_to_request 时负责将两种表达方式（顶层列表 vs content block）
    按目标 provider 的规范进行转换。
    """

    model_config = ConfigDict(populate_by_name=True)

    role: Role
    content: List[ContentBlock] = Field(default_factory=list)
    """多模态内容块列表，按顺序排列。"""

    id: Optional[str] = None
    """消息 ID，OpenAI Responses API / Assistants API 会返回此字段。"""

    name: Optional[str] = None
    """
    发言者名称，OpenAI Chat 的 name 字段（多 agent 场景区分不同 user）。
    """

    tool_calls: List[ToolCall] = Field(default_factory=list)
    """
    工具调用列表（顶层表达，OpenAI Chat / Ollama 风格）。
    assistant 角色消息中，模型请求调用的工具列表。
    与 content 中的 ToolCallBlock 互为补充，适配器按需使用。
    """

    tool_results: List[ToolResult] = Field(default_factory=list)
    """
    工具执行结果列表（顶层表达，OpenAI Chat tool role 风格）。
    tool 角色消息中，应用层填充的工具执行结果。
    与 content 中的 ToolResultBlock 互为补充，适配器按需使用。
    """

    provider: Dict[str, Any] = Field(default_factory=dict)
    """厂商原始数据透传区，例如 {"raw": <原始 message dict>}。"""

    # ------------------------------------------------------------------
    # 快捷构造方法
    # ------------------------------------------------------------------

    @classmethod
    def user_text(cls, text: str) -> Message:
        """快捷构造纯文本 user 消息。"""
        from anyllm.schema.content import TextBlock
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant_text(cls, text: str) -> Message:
        """快捷构造纯文本 assistant 消息。"""
        from anyllm.schema.content import TextBlock
        return cls(role="assistant", content=[TextBlock(text=text)])

    @classmethod
    def tool_result(cls, call_id: str, text: str, is_error: bool = False) -> Message:
        """快捷构造工具结果消息（顶层列表风格）。"""
        from anyllm.schema.content import TextBlock
        return cls(
            role="tool",
            content=[],
            tool_results=[
                ToolResult(
                    call_id=call_id,
                    content=[TextBlock(text=text)],
                    is_error=is_error,
                )
            ],
        )
