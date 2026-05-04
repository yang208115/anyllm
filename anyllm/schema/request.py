"""
UniversalRequest — AnyLLM 统一请求体（PRD §5）。

设计要点：
  - version    : UIR 版本号，用于兼容性检测，当前为 "uai.v1"
  - model      : ModelRef，包含 provider 信息，而不是裸字符串
  - instructions: System Prompt 抽离至顶层，类型为 List[ContentBlock]（支持多模态指令）
  - messages   : 对话历史，含 user/assistant/tool/system/developer 角色
  - state      : 有状态会话信息（thread_id, previous_response_id 等）
  - vendor     : 厂商特定字段透传区，按 provider 名称分组
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from anyllm.schema.content import ContentBlock
from anyllm.schema.message import Message
from anyllm.schema.tools import ResponseFormat, ToolChoice, ToolDef

ProviderName = Literal[
    "openai",
    "anthropic",
    "google",
    "bedrock",
    "ollama",
    "cloudflare",
    "unknown",
]
"""已知 provider 标识符，用于 ModelRef.provider 和 vendor 字段的键名。"""

StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_calls",
    "content_filter",
    "error",
    "unknown",
]
"""统一停止原因枚举（PRD §24），适配器负责将 provider 原始值映射到此枚举。"""


class ModelRef(BaseModel):
    """
    模型引用，包含模型名称和 provider 来源。

    provider 字段用于路由到正确的适配器，name 字段直接传给 provider API。
    raw 字段保留 provider 返回的原始模型信息（例如版本、别名等）。
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    """模型名称，例如 "gpt-4o"、"claude-sonnet-4-5"、"gemini-1.5-pro"。"""

    provider: ProviderName = "unknown"
    """来源 provider，路由适配器时使用。"""

    raw: Dict[str, Any] = Field(default_factory=dict)
    """provider 返回的原始模型元数据透传区。"""


class GenerationConfig(BaseModel):
    """
    生成参数配置（PRD §5.1）。

    适配说明：
      OpenAI Chat : temperature, top_p, max_tokens, stop, presence_penalty, frequency_penalty
      Anthropic   : temperature, top_p, top_k, max_tokens, stop_sequences
      Gemini      : temperature, topP, topK, maxOutputTokens, stopSequences, seed
      Bedrock     : inferenceConfig.{temperature, topP, maxTokens, stopSequences}
    """

    model_config = ConfigDict(populate_by_name=True)

    temperature: Optional[float] = None
    """采样温度，0 = 确定性，值越大越随机。各 provider 上限不同（Anthropic 最大 1）。"""

    top_p: Optional[float] = None
    """Nucleus sampling 参数，通常与 temperature 二选一。"""

    top_k: Optional[int] = None
    """Top-K 采样，Anthropic / Gemini / Bedrock 支持，OpenAI 不支持。"""

    max_output_tokens: Optional[int] = None
    """
    最大输出 token 数。
    Anthropic 要求必填（适配器层默认填 1024），其余 provider 可选。
    映射时注意字段名差异：OpenAI 为 max_tokens，Bedrock 为 maxTokens。
    """

    stop: Optional[List[str]] = None
    """停止序列列表，遇到任一序列时停止生成。"""

    seed: Optional[int] = None
    """随机种子，用于复现输出（OpenAI / Gemini 支持，Anthropic 不支持）。"""

    presence_penalty: Optional[float] = None
    """存在惩罚，OpenAI 专用，其他 provider 忽略。"""

    frequency_penalty: Optional[float] = None
    """频率惩罚，OpenAI 专用，其他 provider 忽略。"""

    raw: Dict[str, Any] = Field(default_factory=dict)
    """额外的生成参数透传区，用于传递 UIR 未覆盖的厂商特定参数。"""


class ConversationState(BaseModel):
    """
    有状态会话信息（PRD §5.1，L2 能力）。

    不同 provider 的状态标识符命名不同，UIR 统一保留所有变体：
      OpenAI Responses  : previous_response_id, conversation_id
      OpenAI Assistants : thread_id, run_id, provider_state.assistant_id
      Anthropic / Gemini / Bedrock / Ollama : 均不支持服务端状态，
        适配器遇到非空 state 时需发出 STATE_NOT_SUPPORTED warning。
    """

    model_config = ConfigDict(populate_by_name=True)

    conversation_id: Optional[str] = None
    """OpenAI Responses API 会话 ID。"""

    thread_id: Optional[str] = None
    """OpenAI Assistants API Thread ID。"""

    run_id: Optional[str] = None
    """OpenAI Assistants API Run ID。"""

    previous_response_id: Optional[str] = None
    """OpenAI Responses API 上一轮响应 ID，用于继续有状态对话。"""

    provider_state: Dict[str, Any] = Field(default_factory=dict)
    """
    其他 provider 特定状态数据，例如：
    {"assistant_id": "asst_xxx"}  # OpenAI Assistants
    """


class UniversalRequest(BaseModel):
    """
    AnyLLM 统一请求体，网关/转换器的唯一内部流通格式。

    完整字段说明见 PRD §5.1。

    使用示例：
        request = UniversalRequest(
            version="uai.v1",
            model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
            instructions=[TextBlock(text="你是一个天气助手。")],
            messages=[
                Message.user_text("东京明天天气怎么样？"),
            ],
            tools=[ToolDef(name="get_weather", description="获取天气", ...)],
            generation=GenerationConfig(temperature=0.7, max_output_tokens=1024),
        )
    """

    model_config = ConfigDict(populate_by_name=True)

    version: str = "uai.v1"
    """UIR 版本号，当前固定为 "uai.v1"，用于兼容性检测。"""

    model: ModelRef
    """目标模型引用，包含 provider 和 name。"""

    messages: List[Message]
    """对话历史，按时间顺序排列。"""

    instructions: List[ContentBlock] = Field(default_factory=list)
    """
    System Prompt，抽离至顶层，类型为 List[ContentBlock] 以支持多模态指令。
    通常只包含 TextBlock，但理论上可包含 ImageBlock（如 vision 系统提示）。

    适配器映射规则：
      OpenAI Chat      : 转为首条 {"role": "system", "content": str}
      OpenAI Responses : 转为 instructions 字段（纯文本）
      Anthropic        : 转为顶层 system 字段
      Gemini           : 转为 systemInstruction.parts
      Bedrock          : 转为顶层 system[{text: ...}]
    """

    tools: List[ToolDef] = Field(default_factory=list)
    """可供模型调用的工具列表，为空时禁用 function calling。"""

    tool_choice: Optional[ToolChoice] = None
    """工具选择策略，None 时由适配器决定默认行为（通常等同于 auto）。"""

    response_format: Optional[ResponseFormat] = None
    """结构化输出格式，None 时使用纯文本输出。"""

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    """生成参数，温度、最大 token 等。"""

    stream: bool = False
    """是否启用流式响应。"""

    state: ConversationState = Field(default_factory=ConversationState)
    """有状态会话信息，L2 能力，大多数 provider 不支持。"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """用户自定义元数据，不传给 provider，用于网关层内部追踪。"""

    vendor: Dict[str, Any] = Field(default_factory=dict)
    """
    厂商特定字段透传区，按 provider 名称分组，例如：
    {
        "anthropic": {"thinking": {"type": "enabled", "budget_tokens": 10000}},
        "openai": {"store": True},
    }
    适配器在 uir_to_request 时将对应的 vendor[provider_name] 合并到输出 dict。
    """
