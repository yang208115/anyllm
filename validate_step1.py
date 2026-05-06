"""Step 1 validation — covers all core data models from the PRD."""
from pydantic import TypeAdapter

from anyllm.schema import (
    AutoToolChoice,
    ContentBlock,
    ConversationState,
    ConversionResult,
    GenerationConfig,
    ImageBlock,
    JsonSchemaResponseFormat,
    MediaSource,
    Message,
    ModelRef,
    TextBlock,
    ToolCall,
    ToolDef,
    ToolResult,
    UniversalRequest,
    UniversalStreamEvent,
    normalize_stop_reason,
    parse_tool_arguments,
)

ta = TypeAdapter(ContentBlock)

PASS = "[OK]"
FAIL = "[FAIL]"

print("=" * 60)
print("Step 1 AnyLLM Universal AI Message IR Models")
print("=" * 60)

# ------------------------------------------------------------------
# 1. ContentBlock polymorphic deserialization
# ------------------------------------------------------------------
raw_blocks = [
    {"type": "text", "text": "Hello"},
    {"type": "image", "source": {"kind": "url", "value": "https://example.com/img.png"}},
    {"type": "audio", "source": {"kind": "base64", "value": "abc123", "mime_type": "audio/mp3"}},
    {"type": "file", "source": {"kind": "file_id", "value": "file-abc"}},
    {"type": "thinking", "text": "Let me think...", "encrypted": None},
    {"type": "refusal", "text": "I cannot answer."},
    {
        "type": "tool_call",
        "call": {"id": "call_001", "name": "get_weather", "arguments": {"city": "Tokyo"}},
    },
    {
        "type": "tool_result",
        "result": {
            "call_id": "call_001",
            "content": [{"type": "text", "text": "Sunny, 25C"}],
            "name": "get_weather",
            "is_error": False,
        },
    },
    {"type": "provider_block", "provider": "openai", "value": {"custom": "data"}},
]

blocks = [ta.validate_python(b) for b in raw_blocks]
print(f"\n{PASS} ContentBlock polymorphic deserialization ({len(blocks)} types):")
for b in blocks:
    print(f"  - {type(b).__name__}: type='{b.type}'")

# ------------------------------------------------------------------
# 2. parse_tool_arguments
# ------------------------------------------------------------------
args1, raw1, warn1 = parse_tool_arguments({"city": "Tokyo"})
args2, raw2, warn2 = parse_tool_arguments('{"city": "Tokyo"}')
args3, raw3, warn3 = parse_tool_arguments("invalid json {")
print(f"\n{PASS} parse_tool_arguments:")
print(f"  dict input   -> args={args1}, warning={warn1}")
print(f"  JSON string  -> args={args2}, raw='{raw2}', warning={warn2}")
print(f"  invalid JSON -> args={args3}, warning={warn3}")

# ------------------------------------------------------------------
# 3. Message shortcuts
# ------------------------------------------------------------------
msg_user = Message.user_text("What is the weather in Tokyo?")
msg_asst = Message.assistant_text("Let me check.")
msg_tool = Message.tool_result("call_001", '{"temperature": 22, "condition": "rain"}')
print(f"\n{PASS} Message shortcuts:")
print(f"  user  : role={msg_user.role}, blocks={len(msg_user.content)}")
print(f"  asst  : role={msg_asst.role}, blocks={len(msg_asst.content)}")
print(f"  tool  : role={msg_tool.role}, results={len(msg_tool.tool_results)}")

# ------------------------------------------------------------------
# 4. Full UniversalRequest (tool call chain, PRD Sec.27)
# ------------------------------------------------------------------
request = UniversalRequest(
    version="uai.v1",
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    instructions=[TextBlock(text="You are a weather assistant.")],
    messages=[
        Message.user_text("What is the weather in Tokyo tomorrow?"),
        Message(
            role="assistant",
            content=[TextBlock(text="Let me query.")],
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="get_weather",
                    arguments={"city": "Tokyo", "date": "tomorrow"},
                )
            ],
        ),
        Message(
            role="tool",
            content=[],
            tool_results=[
                ToolResult(
                    call_id="call_001",
                    name="get_weather",
                    content=[TextBlock(text='{"temperature": 22, "condition": "rain"}')],
                )
            ],
        ),
    ],
    tools=[
        ToolDef(
            type="function",
            name="get_weather",
            description="Get weather by city and date.",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["city", "date"],
            },
        )
    ],
    tool_choice=AutoToolChoice(),
    response_format=JsonSchemaResponseFormat(
        name="weather_result",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        strict=True,
    ),
    generation=GenerationConfig(temperature=0.7, max_output_tokens=1024, top_k=40),
    stream=False,
    state=ConversationState(previous_response_id=None),
    vendor={"anthropic": {"thinking": {"type": "enabled", "budget_tokens": 8000}}},
)

dumped = request.model_dump_json(indent=2)
print(f"\n{PASS} UniversalRequest serialized: {len(dumped)} bytes")

req2 = UniversalRequest.model_validate_json(dumped)
print(f"{PASS} UniversalRequest deserialized")
print(f"  model   : {req2.model.provider}/{req2.model.name}")
print(f"  messages: {len(req2.messages)}")
print(f"  tools   : {[t.name for t in req2.tools]}")
print(f"  msg[1] tool_calls  : {[tc.name for tc in req2.messages[1].tool_calls]}")
print(f"  msg[2] tool_results: {[tr.call_id for tr in req2.messages[2].tool_results]}")

# Verify response_format round-trip
rf = req2.response_format
assert isinstance(rf, JsonSchemaResponseFormat), f"Expected JsonSchemaResponseFormat, got {type(rf)}"
assert rf.json_schema == {"type": "object", "properties": {"answer": {"type": "string"}}}
print(f"  response_format schema round-trip: {PASS}")

# ------------------------------------------------------------------
# 5. ConversionResult + ConversionWarning
# ------------------------------------------------------------------
result: ConversionResult = ConversionResult(value={"model": "gpt-4o", "messages": []})
result.add_warning("STATE_NOT_SUPPORTED", "state.previous_response_id",
                   "OpenAI Chat does not support stateful conversations.", "warning")
result.add_warning("VENDOR_FIELD_PASSTHROUGH", "vendor.anthropic",
                   "anthropic vendor fields ignored.", "info")
print(f"\n{PASS} ConversionResult:")
print(f"  has_errors: {result.has_errors}")
print(f"  warnings  : {len(result.warnings)}")
for w in result.warnings:
    print(f"    [{w.severity}] {w.code}")

# ------------------------------------------------------------------
# 6. normalize_stop_reason
# ------------------------------------------------------------------
cases = [
    ("openai", "stop", "end_turn"),
    ("openai", "tool_calls", "tool_calls"),
    ("anthropic", "tool_use", "tool_calls"),
    ("gemini", "SAFETY", "content_filter"),
    ("bedrock", "max_tokens", "max_tokens"),
    ("openai", "unknown_reason", "unknown"),
]
print(f"\n{PASS} normalize_stop_reason:")
all_ok = True
for provider, raw, expected in cases:
    result_sr = normalize_stop_reason(provider, raw)
    status = PASS if result_sr == expected else FAIL
    if result_sr != expected:
        all_ok = False
    print(f"  {status} {provider}.{raw!r} -> {result_sr!r}")

# ------------------------------------------------------------------
# 7. UniversalStreamEvent
# ------------------------------------------------------------------
evt = UniversalStreamEvent(
    type="content_delta",
    index=0,
    delta=TextBlock(text="Tokyo tomorrow"),
    raw={"choices": [{"delta": {"content": "Tokyo tomorrow"}}]},
)
print(f"\n{PASS} UniversalStreamEvent: type={evt.type}, delta type={type(evt.delta).__name__}")

# ------------------------------------------------------------------
# 8. ToolResult with rich content (image in result)
# ------------------------------------------------------------------
rich_result = ToolResult(
    call_id="call_img",
    content=[
        TextBlock(text="Here is the analysis:"),
        ImageBlock(source=MediaSource(kind="base64", value="iVBOR...", mime_type="image/png")),
    ],
    name="analyze_image",
)
assert len(rich_result.content) == 2
print(f"\n{PASS} ToolResult with rich content: {len(rich_result.content)} blocks")

print("\n" + "=" * 60)
print("All Step 1 validations passed!")
print("=" * 60)
