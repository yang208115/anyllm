"""AnyLLM — 通用大模型网关核心库

UIR (Universal AI Message IR) 版本: uai.v1
支持 provider: OpenAI Chat, OpenAI Responses, OpenAI Assistants,
               Anthropic, Google Gemini, Amazon Bedrock, Ollama, Cloudflare Workers AI
"""

from anyllm.schema import (
    AudioBlock,
    AutoToolChoice,
    ContentBlock,
    ConversionResult,
    ConversionWarning,
    ConversationState,
    FileBlock,
    GenerationConfig,
    ImageBlock,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    MediaSource,
    Message,
    ModelRef,
    NoneToolChoice,
    ProviderBlock,
    ProviderName,
    RefusalBlock,
    RequiredToolChoice,
    ResponseFormat,
    Role,
    SpecificToolChoice,
    StopReason,
    StreamEventType,
    TextBlock,
    TextResponseFormat,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolChoice,
    ToolDef,
    ToolResult,
    ToolResultBlock,
    UniversalRequest,
    UniversalResponse,
    UniversalStreamEvent,
    Usage,
    parse_tool_arguments,
)

from anyllm.adapters.base import BaseAdapter, BaseInterceptor, ProviderCapabilities
from anyllm.adapters.openai_chat import OpenAIChatAdapter
from anyllm.adapters.anthropic import AnthropicAdapter
from anyllm.conversion.converter import UniversalConverter
from anyllm.interceptors import (
    FunctionInterceptor,
    ImageResolutionInterceptor,
    RoleConsolidationInterceptor,
    interceptor,
)
from anyllm.gateway import AnyLLMGateway, GatewayResult, ProviderConfig

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # gateway
    "AnyLLMGateway",
    "GatewayResult",
    "ProviderConfig",
    # converter
    "UniversalConverter",
    # adapters
    "BaseAdapter",
    "BaseInterceptor",
    "ProviderCapabilities",
    "OpenAIChatAdapter",
    "AnthropicAdapter",
    # interceptors
    "ImageResolutionInterceptor",
    "RoleConsolidationInterceptor",
    "FunctionInterceptor",
    "interceptor",
    # content blocks
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
    # request / response
    "ProviderName",
    "StopReason",
    "ModelRef",
    "GenerationConfig",
    "ConversationState",
    "UniversalRequest",
    "UniversalResponse",
    # stream
    "StreamEventType",
    "UniversalStreamEvent",
    # usage
    "Usage",
    # warnings
    "ConversionWarning",
    "ConversionResult",
]
