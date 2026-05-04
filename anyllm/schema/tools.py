"""
ToolDef、ToolChoice、ResponseFormat — 工具定义与输出格式（PRD §8, §9）。

注意：ToolCall / ToolResult 定义在 content.py 中（因与 ContentBlock 存在循环引用）。
本模块只负责 ToolDef（工具声明）、ToolChoice（选择策略）和 ResponseFormat（结构化输出）。
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# ToolDef — 工具定义
# ---------------------------------------------------------------------------

class ToolDef(BaseModel):
    """
    工具/函数声明，遵循 JSON Schema Draft-07 规范。

    type 枚举：
      function : 用户自定义函数（MVP 重点支持）
      builtin  : provider 内置工具（web_search、code_interpreter 等）
      mcp      : Model Context Protocol 工具（L2，后续支持）
      provider : 完全透传给 provider 的私有工具格式

    适配说明：
      OpenAI Chat   : tools[].function {name, description, parameters, strict}
      OpenAI Resp   : tools[] 支持 function / file_search / web_search_preview 等
      Anthropic     : tools[] {name, description, input_schema}
      Gemini        : tools[].functionDeclarations[] {name, description, parameters}
      Bedrock       : toolConfig.tools[].toolSpec {name, description, inputSchema.json}
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["function", "builtin", "mcp", "provider"] = "function"
    name: str
    """工具名称，需唯一，建议 snake_case，不超过 64 个字符。"""

    description: Optional[str] = None
    """工具功能描述，供模型理解何时应调用此工具。尽量简洁清晰。"""

    input_schema: Optional[Dict[str, Any]] = None
    """
    输入参数 JSON Schema，例如：
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    """

    output_schema: Optional[Dict[str, Any]] = None
    """输出结果 JSON Schema（OpenAI Responses strict mode 支持，其他 provider 忽略）。"""

    provider: Dict[str, Any] = Field(default_factory=dict)
    """厂商特定参数透传区，例如 Anthropic cache_control、OpenAI strict。"""


# ---------------------------------------------------------------------------
# ToolChoice — 工具选择策略
# ---------------------------------------------------------------------------

class AutoToolChoice(BaseModel):
    """让模型自动决定是否调用工具。"""
    type: Literal["auto"] = "auto"


class NoneToolChoice(BaseModel):
    """禁止模型调用任何工具。"""
    type: Literal["none"] = "none"


class RequiredToolChoice(BaseModel):
    """
    强制模型必须调用至少一个工具。

    适配说明：
      OpenAI : "required"
      Anthropic : {"type": "any"}
      Gemini : FunctionCallingConfig.Mode.ANY
      Bedrock : {"auto": {}} 或 toolChoice.any
    """
    type: Literal["required"] = "required"


class SpecificToolChoice(BaseModel):
    """
    强制模型调用指定工具。

    适配说明：
      OpenAI : {"type": "function", "function": {"name": ...}}
      Anthropic : {"type": "tool", "name": ...}
      Gemini : FunctionCallingConfig with allowedFunctionNames
      Bedrock : {"tool": {"name": ...}}
    """
    type: Literal["tool"] = "tool"
    name: str
    """必须调用的工具名称。"""


ToolChoice = Annotated[
    Union[AutoToolChoice, NoneToolChoice, RequiredToolChoice, SpecificToolChoice],
    Field(discriminator="type"),
]
"""工具选择策略，支持 "auto" | "none" | "required" | "tool"。"""


# ---------------------------------------------------------------------------
# ResponseFormat — 结构化输出格式（PRD §9）
# ---------------------------------------------------------------------------

class TextResponseFormat(BaseModel):
    """纯文本输出（默认模式）。"""
    type: Literal["text"] = "text"


class JsonObjectResponseFormat(BaseModel):
    """
    JSON 对象输出，不约束 schema。

    适配说明：
      OpenAI : {"type": "json_object"}
      Anthropic : 需要在 system prompt 中说明（不原生支持此格式）
      Gemini : responseMimeType = "application/json"
    """
    type: Literal["json_object"] = "json_object"


class JsonSchemaResponseFormat(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    """
    带 schema 约束的 JSON 输出（Structured Outputs）。

    适配说明：
      OpenAI : {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}
      Anthropic : 通过 system prompt 描述 schema（降级，发出 RESPONSE_FORMAT_DOWNGRADED warning）
      Gemini : responseSchema + responseMimeType
      Bedrock : 通过 system/prompt 描述（降级）
    """
    type: Literal["json_schema"] = "json_schema"
    name: str
    """schema 名称标识符，OpenAI 要求必填。"""

    json_schema: Dict[str, Any] = Field(alias="schema")
    """JSON Schema 定义，描述期望的输出结构。字段名为 'schema'，以保持与 OpenAI API 兼容。"""

    strict: bool = False
    """
    OpenAI Structured Outputs strict mode，要求所有字段声明且无 additionalProperties。
    其他 provider 忽略此字段。
    """


ResponseFormat = Annotated[
    Union[TextResponseFormat, JsonObjectResponseFormat, JsonSchemaResponseFormat],
    Field(discriminator="type"),
]
"""结构化输出格式，支持 "text" | "json_object" | "json_schema"。"""
