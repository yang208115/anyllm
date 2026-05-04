"""
AnyLLM Schema 包 — 统一导出所有 UIR 数据模型。

导入顺序很重要：content.py 末尾已调用 model_rebuild()，
其余模块的前向引用在各自模块内已处理完毕。
"""

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
from anyllm.schema.message import Message, Role
from anyllm.schema.request import (
    ConversationState,
    GenerationConfig,
    ModelRef,
    ProviderName,
    StopReason,
    UniversalRequest,
)
from anyllm.schema.response import (
    PROVIDER_STOP_REASON_MAP,
    UniversalResponse,
    normalize_stop_reason,
)
from anyllm.schema.stream import StreamEventType, UniversalStreamEvent
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

__all__ = [
    # content
    "MediaSource",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "FileBlock",
    "ThinkingBlock",
    "RefusalBlock",
    "ToolCall",
    "ToolResult",
    "ToolCallBlock",
    "ToolResultBlock",
    "ProviderBlock",
    "ContentBlock",
    "parse_tool_arguments",
    # message
    "Role",
    "Message",
    # tools
    "ToolDef",
    "AutoToolChoice",
    "NoneToolChoice",
    "RequiredToolChoice",
    "SpecificToolChoice",
    "ToolChoice",
    "TextResponseFormat",
    "JsonObjectResponseFormat",
    "JsonSchemaResponseFormat",
    "ResponseFormat",
    # request
    "ProviderName",
    "StopReason",
    "ModelRef",
    "GenerationConfig",
    "ConversationState",
    "UniversalRequest",
    # response
    "UniversalResponse",
    "PROVIDER_STOP_REASON_MAP",
    "normalize_stop_reason",
    # stream
    "StreamEventType",
    "UniversalStreamEvent",
    # usage
    "Usage",
    # warnings
    "ConversionWarning",
    "ConversionResult",
]
