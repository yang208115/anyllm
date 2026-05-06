"""
Provider 能力矩阵预设值（PRD §12）。

每个预设实例描述了对应 provider / API 接口的能力范围。
适配器类通过 capabilities 属性返回对应的预设，
UniversalConverter / Gateway 据此做转换前的兼容性检查。

注意：
  - 某些 provider（如 Ollama、Cloudflare）的能力取决于具体部署的模型，
    这里给出的是保守估计（default safe）。
  - 适配器子类可以覆盖 capabilities 属性以提供更精确的运行时能力声明。
"""

from anyllm.adapters.base import ProviderCapabilities

# =====================================================================
# OpenAI 系列
# =====================================================================

OPENAI_CHAT_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,
    streaming=True,
    # L1
    image_input=True,           # GPT-4o / GPT-4 Vision 支持
    audio_input=False,          # Chat Completions 暂无音频输入
    file_input=False,           # 需要通过 Assistants / Responses API
    tools=True,
    parallel_tool_calls=True,   # 支持一次返回多个 tool_calls
    tool_result_blocks=False,   # tool result 是独立的 tool 角色消息，不是 content block
    json_object=True,
    json_schema=True,           # Structured Outputs (gpt-4o-2024-08-06+)
    # L2
    stateful=False,             # Chat Completions 无服务端状态
    previous_response_id=False,
    builtin_tools=False,        # 无内置工具
    developer_role=True,        # 支持 developer 角色
)

OPENAI_RESPONSES_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,
    streaming=True,
    # L1
    image_input=True,
    audio_input=False,
    file_input=True,            # 支持 file 类型 input item
    tools=True,
    parallel_tool_calls=True,
    tool_result_blocks=False,
    json_object=True,
    json_schema=True,
    # L2
    stateful=True,              # 支持 previous_response_id + conversation
    previous_response_id=True,
    builtin_tools=True,         # web_search_preview, file_search, code_interpreter
    developer_role=True,
)

OPENAI_ASSISTANTS_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,
    streaming=True,
    # L1
    image_input=True,
    audio_input=False,
    file_input=True,
    tools=True,
    parallel_tool_calls=True,
    tool_result_blocks=False,
    json_object=True,
    json_schema=True,
    # L2
    stateful=True,              # thread + run 模型
    previous_response_id=False,
    builtin_tools=True,         # file_search, code_interpreter
    developer_role=False,
)


# =====================================================================
# Anthropic
# =====================================================================

ANTHROPIC_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,    # 顶层 system 字段
    streaming=True,
    # L1
    image_input=True,           # 仅 base64，需要 ImageResolutionInterceptor
    audio_input=False,
    file_input=False,           # Anthropic 原生不支持文件引用
    tools=True,
    parallel_tool_calls=True,   # 一次可返回多个 tool_use block
    tool_result_blocks=True,    # tool_result 作为 user 消息的 content block
    json_object=False,          # 需要 prompt 工程，不原生支持
    json_schema=False,          # 同上
    # L2
    stateful=False,
    previous_response_id=False,
    builtin_tools=False,        # computer_use 等实验性工具，暂不纳入
    developer_role=False,
)


# =====================================================================
# Google Gemini
# =====================================================================

GEMINI_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,    # systemInstruction 字段
    streaming=True,
    # L1
    image_input=True,           # inlineData / fileData
    audio_input=True,           # Gemini 支持音频输入
    file_input=True,            # 通过 File API 上传
    tools=True,
    parallel_tool_calls=True,
    tool_result_blocks=True,    # functionResponse 作为 part
    json_object=True,           # responseMimeType: application/json
    json_schema=True,           # responseSchema
    # L2
    stateful=False,
    previous_response_id=False,
    builtin_tools=False,        # code_execution 等实验性工具
    developer_role=False,
)


# =====================================================================
# Amazon Bedrock (Converse API)
# =====================================================================

BEDROCK_CONVERSE_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,    # 顶层 system 字段
    streaming=True,
    # L1
    image_input=True,           # image.source.bytes
    audio_input=False,
    file_input=False,
    tools=True,
    parallel_tool_calls=True,
    tool_result_blocks=True,    # toolResult 在 content 中
    json_object=False,
    json_schema=False,
    # L2
    stateful=False,
    previous_response_id=False,
    builtin_tools=False,
    developer_role=False,
)


# =====================================================================
# Ollama (OpenAI-compatible 模式)
# =====================================================================

OLLAMA_OPENAI_COMPAT_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,
    streaming=True,
    # L1 — 取决于具体模型，这里给出保守估计
    image_input=False,          # 部分 vision 模型支持，默认关闭
    audio_input=False,
    file_input=False,
    tools=True,                 # Ollama 支持 tool calling
    parallel_tool_calls=False,  # 保守估计
    tool_result_blocks=False,
    json_object=True,           # 部分模型支持
    json_schema=False,          # 保守估计
    # L2
    stateful=False,
    previous_response_id=False,
    builtin_tools=False,
    developer_role=False,
)


# =====================================================================
# Cloudflare Workers AI (OpenAI-compatible 模式)
# =====================================================================

CLOUDFLARE_WORKERS_AI_CAPABILITIES = ProviderCapabilities(
    # L0
    text=True,
    system_instruction=True,
    streaming=True,
    # L1 — 取决于部署的模型
    image_input=True,           # 部分模型支持
    audio_input=False,
    file_input=False,
    tools=True,                 # 部分模型支持
    parallel_tool_calls=False,
    tool_result_blocks=False,
    json_object=True,           # 部分模型支持
    json_schema=True,           # 部分模型支持
    # L2
    stateful=False,
    previous_response_id=False,
    builtin_tools=False,
    developer_role=False,
)
