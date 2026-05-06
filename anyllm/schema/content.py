"""
ContentBlock 体系 — AnyLLM 最小内容单元（PRD §7）。

设计原则（原则三：content 永远是 block array）：
  所有消息内容统一表示为 List[ContentBlock]，
  不使用 content: str 的简化格式，以便在所有 provider 间无损转换。

Block 类型一览：
  text          纯文本
  image         图片（url / base64 / file_id / bytes）
  audio         音频（base64 / file_id）
  file          文件引用（file_id / bytes）
  thinking      推理过程（Anthropic extended thinking / OpenAI reasoning）
  refusal       模型拒绝响应（OpenAI refusal block）
  tool_call     模型请求调用工具（wraps ToolCall）
  tool_result   工具执行结果（wraps ToolResult）
  provider_block 厂商私有块，透传保留

循环依赖处理：
  ToolResult.content: List[ContentBlock] 与 ToolResultBlock.result: ToolResult
  构成循环引用。解决方案：将 ToolCall / ToolResult 定义在本模块，
  使用 from __future__ import annotations 使所有注解为字符串，
  在模块末尾调用 model_rebuild() 解析前向引用。
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# MediaSource — 统一媒体来源描述符
# ---------------------------------------------------------------------------

class MediaSource(BaseModel):
    """
    统一媒体来源，支持四种 kind：
      url     : 公网可访问的 HTTP(S) URL（OpenAI / Gemini 原生支持）
      base64  : Base64 编码的原始字节，不含 data URI 前缀
      file_id : 已上传到平台的文件 ID（OpenAI Files API、Gemini File API）
      bytes   : 原始字节对象（Bedrock Converse 的 image.source.bytes）

    注意：Anthropic 仅支持 base64；Bedrock 图片通常用 bytes；
          ImageResolutionInterceptor 负责将 url 自动转换为 base64/bytes。
    """

    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["url", "base64", "file_id", "bytes"]
    value: Any
    """
    实际数据：
      url     → str（URL 字符串）
      base64  → str（Base64 编码，不含 data: 前缀）
      file_id → str（平台文件 ID）
      bytes   → bytes（原始字节）
    """

    mime_type: str | None = None
    """MIME 类型，例如 "image/png"，base64 / bytes 时通常必填。"""

    filename: str | None = None
    """文件名，file_id / bytes 时可选，辅助 provider 识别格式。"""


# ---------------------------------------------------------------------------
# 基础媒体 Block（无循环依赖）
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    """
    纯文本块。

    适配说明：
      OpenAI Chat/Responses : {"type": "text", "text": ...}
      Anthropic             : {"type": "text", "text": ...}
      Gemini                : Part.text
      Bedrock Converse      : {"text": ...}
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["text"] = "text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    """OpenAI Responses API 返回的 annotations（引用、URL 等），其他 provider 忽略。"""


class ImageBlock(BaseModel):
    """
    图片块，支持 url / base64 / file_id 三种来源。

    适配说明：
      OpenAI Chat   : url → image_url; base64 → data URI in image_url.url
      OpenAI Resp   : url → image_url; base64 → image_file(file_id) or data URI
      Anthropic     : 仅 base64 → {"type":"image","source":{"type":"base64",...}}
                      url 需 ImageResolutionInterceptor 预先转换
      Gemini        : base64 → inlineData; url → fileData
      Bedrock       : bytes → {"image":{"format":...,"source":{"bytes":...}}}
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["image"] = "image"
    source: MediaSource
    detail: Literal["low", "high", "auto"] | None = None
    """图片分辨率控制，OpenAI Vision 专用；其他 provider 适配器忽略。"""


class AudioBlock(BaseModel):
    """
    音频块（L2 能力，MVP 后实现）。

    适配说明：
      OpenAI Responses : input_audio block
      Gemini           : Part.inline_data（audio mime）
      其他 provider    : 通常不支持，适配器需发出 UNSUPPORTED_MODALITY warning
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["audio"] = "audio"
    source: MediaSource
    format: str | None = None
    """音频编码格式，例如 "mp3", "wav", "opus"。"""


class FileBlock(BaseModel):
    """
    文件引用块（文档、PDF 等）。

    适配说明：
      OpenAI Responses : file 类型 input item
      Gemini           : Part.file_data
      其他 provider    : 适配器发出 FILE_REFERENCE_NOT_SUPPORTED warning
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["file"] = "file"
    source: MediaSource
    mime_type: str | None = None
    filename: str | None = None


class ThinkingBlock(BaseModel):
    """
    推理/思考过程块。

    适配说明：
      Anthropic extended thinking : {"type":"thinking","thinking":...} 或加密形式
      OpenAI o1/o3 reasoning      : summary block（来自 Responses API）
      其他 provider               : 通常忽略或以 ProviderBlock 透传
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["thinking"] = "thinking"
    text: str | None = None
    """明文推理内容（用于展示）。"""

    encrypted: str | None = None
    """Anthropic 加密形式的 thinking，需原样回传给下一轮请求。"""

    signature: str | None = None
    """Anthropic 用于验证 encrypted thinking 完整性的签名。"""


class RefusalBlock(BaseModel):
    """
    模型拒绝响应块（OpenAI 专用）。

    当模型因安全策略拒绝回答时，OpenAI 会在 content 中插入 refusal block。
    其他 provider 无此概念，适配器可将其转为 TextBlock 或发出 warning。
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["refusal"] = "refusal"
    text: str
    """拒绝原因文本。"""


class ProviderBlock(BaseModel):
    """
    厂商私有块 — 透传保留区。

    当遇到无法识别的 provider-specific content part 时，
    适配器将其包装为 ProviderBlock 而非静默丢弃，
    并附加 VENDOR_FIELD_PASSTHROUGH warning。
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["provider_block"] = "provider_block"
    provider: str
    """来源 provider 标识符，例如 "openai"、"anthropic"。"""

    value: Any
    """原始数据，原样保留。"""


# ---------------------------------------------------------------------------
# ToolCall — 工具调用描述符
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """
    工具调用记录，由模型在 assistant 消息中生成。

    存放位置（取决于 provider 和表达方式）：
      Message.tool_calls[]         : 顶层列表（OpenAI Chat / Ollama 风格）
      ToolCallBlock.call            : 内嵌在 content block 中（Anthropic 风格）
      Message.content[ToolCallBlock]: 同上

    arguments 规范化：
      provider 返回字符串 JSON → 解析为 dict，原始串保留在 raw_arguments
      解析失败 → arguments={}, raw_arguments=原串, 触发 INVALID_TOOL_ARGUMENTS_JSON warning
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    """工具调用唯一 ID，由模型生成，用于匹配 ToolResult.call_id。"""

    name: str
    """工具名称，需与 ToolDef.name 对应。"""

    arguments: Any
    """
    已解析的工具参数（dict）。
    若 provider 返回 JSON 字符串，适配器负责在 request_to_uir 阶段解析。
    """

    raw_arguments: str | None = None
    """provider 返回的原始 arguments 字符串，解析失败时用于调试。"""

    provider: dict[str, Any] = Field(default_factory=dict)
    """厂商原始数据透传区，例如 {"raw": <原始 tool_call dict>}。"""


# ---------------------------------------------------------------------------
# ToolResult — 工具执行结果
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """
    工具执行结果，由调用方（应用层）填充后放入下一轮请求。

    存放位置：
      Message.tool_results[]        : 顶层列表
      ToolResultBlock.result         : 内嵌在 content block 中

    content 可以是纯文本 TextBlock，也可以是包含图片的富文本列表，
    适配器在输出时负责将其降级为目标 provider 支持的格式。
    """

    model_config = ConfigDict(populate_by_name=True)

    call_id: str
    """对应 ToolCall.id，用于关联调用与结果。"""

    content: list[ContentBlock]
    """
    工具结果内容，通常为 [TextBlock(text="...")]，
    也可包含 ImageBlock（富文本结果）。
    ToolResultBlock / ToolResult 不允许嵌套 ToolCallBlock / ToolResultBlock。
    """

    name: str | None = None
    """工具名称，Gemini functionResponse 需要此字段。"""

    is_error: bool = False
    """标记工具执行是否出错，Anthropic tool_result 原生支持此字段。"""

    provider: dict[str, Any] = Field(default_factory=dict)
    """厂商原始数据透传区。"""


# ---------------------------------------------------------------------------
# ToolCallBlock / ToolResultBlock — 内嵌工具 Block
# ---------------------------------------------------------------------------

class ToolCallBlock(BaseModel):
    """
    工具调用内嵌 Block（Anthropic / Gemini / Bedrock 风格）。

    适配说明：
      Anthropic : {"type": "tool_use", "id": ..., "name": ..., "input": ...}
      Gemini    : Part.functionCall
      Bedrock   : {"toolUse": {...}}
      OpenAI    : 不使用此 Block，tool_calls 放在 Message.tool_calls 顶层列表；
                  适配器在 uir_to_request 时将 ToolCallBlock 上移至 tool_calls。
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["tool_call"] = "tool_call"
    call: ToolCall


class ToolResultBlock(BaseModel):
    """
    工具结果内嵌 Block（Anthropic / Gemini / Bedrock 风格）。

    适配说明：
      Anthropic : 必须作为 user 角色消息的 content 元素：
                  {"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": ...}
      Gemini    : Part.functionResponse
      Bedrock   : {"toolResult": {...}}
      OpenAI    : 不使用此 Block，tool 角色消息直接携带 tool_call_id + content。
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["tool_result"] = "tool_result"
    result: ToolResult


# ---------------------------------------------------------------------------
# 顶层多态 ContentBlock 联合类型
# ---------------------------------------------------------------------------

ContentBlock = Annotated[
    TextBlock | ImageBlock | AudioBlock | FileBlock | ThinkingBlock | RefusalBlock | ToolCallBlock | ToolResultBlock | ProviderBlock,
    Field(discriminator="type"),
]
"""
AnyLLM 统一内容块类型，通过 Pydantic v2 Discriminator 实现多态反序列化。
根据 type 字段自动路由到对应子类，无需手动 isinstance 判断。

合法 type 值：
  "text" | "image" | "audio" | "file" | "thinking" |
  "refusal" | "tool_call" | "tool_result" | "provider_block"
"""


# ---------------------------------------------------------------------------
# 解析循环引用（ToolResult.content: List[ContentBlock]）
# ---------------------------------------------------------------------------

ToolResult.model_rebuild()
ToolCallBlock.model_rebuild()
ToolResultBlock.model_rebuild()


# ---------------------------------------------------------------------------
# 工具函数：parse_tool_arguments（PRD §14.2）
# ---------------------------------------------------------------------------

def parse_tool_arguments(
    value: Any,
) -> tuple[Any, str | None, str | None]:
    """
    规范化 tool_call arguments。

    Returns:
        (arguments: dict|Any, raw_arguments: str|None, warning_code: str|None)
        warning_code 为 None 表示解析成功，否则返回 "INVALID_TOOL_ARGUMENTS_JSON"。
    """
    if isinstance(value, dict):
        return value, None, None

    if isinstance(value, str):
        try:
            return json.loads(value), value, None
        except json.JSONDecodeError:
            return {}, value, "INVALID_TOOL_ARGUMENTS_JSON"

    return value, None, None
