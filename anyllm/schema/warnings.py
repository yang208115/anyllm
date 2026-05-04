"""
ConversionWarning 和 ConversionResult — 转换结果包装器。

设计原则（来自 PRD §10）：
  所有适配器的转换函数都不能只返回 dict，必须返回 ConversionResult，
  以便调用方感知哪些字段发生了降级（Degradation）或不兼容。

典型 warning code：
  UNSUPPORTED_ROLE            角色不被目标 provider 支持
  UNSUPPORTED_MODALITY        模态不被目标 provider 支持
  STATE_NOT_SUPPORTED         有状态字段（previous_response_id 等）无法映射
  TOOL_CHOICE_DOWNGRADED      tool_choice 枚举降级
  RESPONSE_FORMAT_DOWNGRADED  结构化输出格式降级
  VENDOR_FIELD_PASSTHROUGH    透传的厂商私有字段
  INVALID_TOOL_ARGUMENTS_JSON tool_call arguments 不是合法 JSON
  STREAM_EVENT_DOWNGRADED     流式事件降级
  BUILTIN_TOOL_NOT_SUPPORTED  内置工具不被支持
  FILE_REFERENCE_NOT_SUPPORTED 文件引用不被支持
  ROLE_DOWNGRADED             角色被合并/降级处理
  IMAGE_SOURCE_DOWNGRADED     图片源格式不被支持，降级为占位符
  API_DEPRECATED_OR_MIGRATING API 即将弃用
  TOOL_CALL_BLOCK_DOWNGRADED  ToolCallBlock 降级为纯文本
  TOOL_RESULT_BLOCK_DOWNGRADED ToolResultBlock 降级为纯文本
  UNSUPPORTED_CONTENT_BLOCK   不支持的内容块类型
"""

from __future__ import annotations

from typing import Any, Generic, List, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ConversionWarning(BaseModel):
    """单条转换警告，记录字段路径、错误码和严重程度。"""

    model_config = ConfigDict(populate_by_name=True)

    code: str
    """机器可读的错误码，见模块 docstring 中的典型值列表。"""

    path: str
    """触发警告的字段路径，例如 "messages[2].content[0]"。"""

    message: str
    """人类可读的描述信息。"""

    severity: Literal["info", "warning", "error"] = "warning"
    """
    严重程度：
      info    : 纯透传/提示，不影响功能。
      warning : 字段降级，目标请求可能丢失部分语义。
      error   : 无法正确转换，目标请求可能不可用。
    """


class ConversionResult(BaseModel, Generic[T]):
    """
    所有适配器转换方法的统一返回类型。

    不要直接返回 dict 或模型对象，必须通过此类包装，
    以便调用方（UniversalConverter / Gateway）统一收集并透传 warnings。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    value: T
    """转换结果，可以是 UniversalRequest、UniversalResponse 或 dict。"""

    warnings: List[ConversionWarning] = Field(default_factory=list)
    """转换过程中产生的所有警告，按发生顺序排列。"""

    def add_warning(
        self,
        code: str,
        path: str,
        message: str,
        severity: Literal["info", "warning", "error"] = "warning",
    ) -> None:
        """便捷方法：追加一条警告，避免重复构造 ConversionWarning。"""
        self.warnings.append(
            ConversionWarning(code=code, path=path, message=message, severity=severity)
        )

    def merge_warnings(self, other: ConversionResult[Any]) -> None:
        """将另一个 ConversionResult 的 warnings 合并进来（用于多步转换链）。"""
        self.warnings.extend(other.warnings)

    @property
    def has_errors(self) -> bool:
        """是否包含 severity=error 级别的警告。"""
        return any(w.severity == "error" for w in self.warnings)
