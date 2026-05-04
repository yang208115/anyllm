# -*- coding: utf-8 -*-
"""Step 4 validation - OpenAI Chat + Anthropic adapters full round-trip."""
import asyncio
import json

from anyllm.adapters.openai_chat import OpenAIChatAdapter
from anyllm.adapters.anthropic import AnthropicAdapter
from anyllm.conversion.converter import UniversalConverter
from anyllm.interceptors import ImageResolutionInterceptor, RoleConsolidationInterceptor
from anyllm.schema import (
    TextBlock, ImageBlock, MediaSource, ToolCall, ToolResult,
    ToolCallBlock, ToolResultBlock,
    Message, ModelRef, GenerationConfig, UniversalRequest,
    AutoToolChoice, SpecificToolChoice,
    JsonSchemaResponseFormat, ToolDef,
)

PASS = "[OK]"
FAIL = "[FAIL]"

print("=" * 60)
print("Step 4: OpenAI Chat + Anthropic Adapters")
print("=" * 60)

openai_adapter = OpenAIChatAdapter()
anthropic_adapter = AnthropicAdapter()

# ==================================================================
# Test 1: OpenAI Chat request_to_uir (basic text + system)
# ==================================================================
print("\n--- 1. OpenAI Chat request_to_uir ---")

openai_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "Thanks!"},
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": False,
}

result = openai_adapter.request_to_uir(openai_request)
uir = result.value
assert uir.model.provider == "openai"
assert uir.model.name == "gpt-4o"
assert len(uir.instructions) == 1  # system message hoisted
assert uir.instructions[0].text == "You are a helpful assistant."
assert len(uir.messages) == 3  # user, assistant, user (no system)
assert uir.generation.temperature == 0.7
assert uir.generation.max_output_tokens == 1024
print(f"{PASS} Basic text + system: {len(uir.instructions)} instr, {len(uir.messages)} msgs")

# ==================================================================
# Test 2: OpenAI Chat request_to_uir (multimodal + tool_calls)
# ==================================================================
print("\n--- 2. OpenAI Chat multimodal + tool_calls ---")

openai_tool_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You analyze images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg", "detail": "high"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "analyze_image",
                        "arguments": '{"detail": "high"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "The image shows a cat.",
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "analyze_image",
                "description": "Analyze image content",
                "parameters": {
                    "type": "object",
                    "properties": {"detail": {"type": "string"}},
                },
            },
        }
    ],
    "tool_choice": "auto",
}

result2 = openai_adapter.request_to_uir(openai_tool_request)
uir2 = result2.value

# Check multimodal user message
user_msg = uir2.messages[0]
assert len(user_msg.content) == 2
assert isinstance(user_msg.content[0], TextBlock)
assert isinstance(user_msg.content[1], ImageBlock)
assert user_msg.content[1].source.kind == "url"
assert user_msg.content[1].detail == "high"
print(f"{PASS} Multimodal user: {len(user_msg.content)} blocks (text+image)")

# Check assistant tool_calls
asst_msg = uir2.messages[1]
assert len(asst_msg.tool_calls) == 1
assert asst_msg.tool_calls[0].name == "analyze_image"
assert asst_msg.tool_calls[0].arguments == {"detail": "high"}
print(f"{PASS} Assistant tool_calls: {asst_msg.tool_calls[0].name}")

# Check tool result message
tool_msg = uir2.messages[2]
assert tool_msg.role == "tool"
assert len(tool_msg.tool_results) == 1
assert tool_msg.tool_results[0].call_id == "call_abc123"
print(f"{PASS} Tool result: call_id={tool_msg.tool_results[0].call_id}")

# Check tools
assert len(uir2.tools) == 1
assert uir2.tools[0].name == "analyze_image"
print(f"{PASS} Tools: {[t.name for t in uir2.tools]}")

# ==================================================================
# Test 3: OpenAI Chat uir_to_request (round-trip)
# ==================================================================
print("\n--- 3. OpenAI Chat uir_to_request (round-trip) ---")

result3 = openai_adapter.uir_to_request(uir2)
out = result3.value
assert out["model"] == "gpt-4o"
assert out["messages"][0]["role"] == "system"
assert out["messages"][0]["content"] == "You analyze images."

# Check multimodal content
user_out = out["messages"][1]
assert isinstance(user_out["content"], list)
assert user_out["content"][0]["type"] == "text"
assert user_out["content"][1]["type"] == "image_url"
assert user_out["content"][1]["image_url"]["url"] == "https://example.com/cat.jpg"
print(f"{PASS} Round-trip: multimodal user content preserved")

# Check tool_calls
asst_out = out["messages"][2]
assert len(asst_out["tool_calls"]) == 1
assert asst_out["tool_calls"][0]["function"]["name"] == "analyze_image"
# arguments should be JSON string
assert json.loads(asst_out["tool_calls"][0]["function"]["arguments"]) == {"detail": "high"}
print(f"{PASS} Round-trip: tool_calls preserved (JSON string)")

# Check tool result
tool_out = out["messages"][3]
assert tool_out["role"] == "tool"
assert tool_out["tool_call_id"] == "call_abc123"
print(f"{PASS} Round-trip: tool result preserved")

# Check tools
assert len(out["tools"]) == 1
assert out["tools"][0]["function"]["name"] == "analyze_image"
print(f"{PASS} Round-trip: tools definition preserved")

# ==================================================================
# Test 4: OpenAI Chat response_to_uir
# ==================================================================
print("\n--- 4. OpenAI Chat response_to_uir ---")

openai_response = {
    "id": "chatcmpl-xxx",
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}

result4 = openai_adapter.response_to_uir(openai_response)
resp = result4.value
assert resp.id == "chatcmpl-xxx"
assert resp.model == "gpt-4o-2024-08-06"
assert resp.stop_reason == "end_turn"
assert resp.usage.input_tokens == 10
assert resp.usage.output_tokens == 8
assert len(resp.output) == 1
assert resp.output[0].content[0].text == "Hello! How can I help you?"
print(f"{PASS} Response: id={resp.id}, stop={resp.stop_reason}, usage={resp.usage.total_tokens}t")

# ==================================================================
# Test 5: Anthropic request_to_uir
# ==================================================================
print("\n--- 5. Anthropic request_to_uir ---")

anthropic_request = {
    "model": "claude-sonnet-4-5-20241022",
    "max_tokens": 1024,
    "system": "You are a weather assistant.",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Tokyo weather?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_001",
                    "content": "Sunny, 25C",
                    "is_error": False,
                }
            ],
        },
    ],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get weather info",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
}

result5 = anthropic_adapter.request_to_uir(anthropic_request)
uir5 = result5.value
assert uir5.model.provider == "anthropic"
assert uir5.model.name == "claude-sonnet-4-5-20241022"
assert len(uir5.instructions) == 1
assert uir5.instructions[0].text == "You are a weather assistant."

# assistant message should have tool_calls
asst5 = uir5.messages[1]
assert len(asst5.tool_calls) == 1
assert asst5.tool_calls[0].name == "get_weather"
print(f"{PASS} Anthropic request_to_uir: model={uir5.model.name}, tool={asst5.tool_calls[0].name}")

# user message with tool_result
user5 = uir5.messages[2]
assert len(user5.tool_results) == 1
assert user5.tool_results[0].call_id == "toolu_001"
print(f"{PASS} Tool result parsed: call_id={user5.tool_results[0].call_id}")

# ==================================================================
# Test 6: Anthropic uir_to_request (round-trip)
# ==================================================================
print("\n--- 6. Anthropic uir_to_request (round-trip) ---")

result6 = anthropic_adapter.uir_to_request(uir5)
out6 = result6.value
assert out6["model"] == "claude-sonnet-4-5-20241022"
assert out6["max_tokens"] == 1024
assert out6["system"] == "You are a weather assistant."

# Check assistant message has tool_use block
asst_out6 = out6["messages"][1]
tool_use_blocks = [b for b in asst_out6["content"] if b.get("type") == "tool_use"]
assert len(tool_use_blocks) == 1
assert tool_use_blocks[0]["name"] == "get_weather"
print(f"{PASS} Anthropic round-trip: tool_use preserved")

# Check tool_result block in user message
user_out6 = out6["messages"][2]
tool_result_blocks = [b for b in user_out6["content"] if b.get("type") == "tool_result"]
assert len(tool_result_blocks) == 1
assert tool_result_blocks[0]["tool_use_id"] == "toolu_001"
print(f"{PASS} Anthropic round-trip: tool_result preserved")

# ==================================================================
# Test 7: Cross-provider conversion OpenAI -> Anthropic
# ==================================================================
print("\n--- 7. Cross-provider: OpenAI Chat -> Anthropic ---")

async def test_cross_provider():
    converter = UniversalConverter()
    converter.register_adapter("openai_chat", openai_adapter)
    converter.register_adapter("anthropic", anthropic_adapter)
    converter.register_interceptor(ImageResolutionInterceptor())
    converter.register_interceptor(RoleConsolidationInterceptor())

    # Simple text request: OpenAI -> Anthropic
    openai_simple = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ],
        "temperature": 0.5,
        "max_tokens": 512,
    }

    result = await converter.convert_request(
        source_provider="openai_chat",
        target_provider="anthropic",
        raw_request=openai_simple,
    )
    out = result.value

    assert out["model"] == "gpt-4o"
    assert out["system"] == "Be concise."
    assert out["max_tokens"] == 512
    assert out["temperature"] == 0.5

    # Messages should only be user/assistant (no system)
    roles = [m["role"] for m in out["messages"]]
    assert "system" not in roles, f"system should not be in messages: {roles}"
    assert roles == ["user", "assistant", "user"], f"Expected alternating roles, got {roles}"
    print(f"  {PASS} OpenAI->Anthropic text: roles={roles}")

    # Check warnings
    print(f"  {PASS} Warnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"    [{w.severity}] {w.code}: {w.message[:60]}")

    return True

asyncio.run(test_cross_provider())

# ==================================================================
# Test 8: Anthropic response_to_uir (with tool_use)
# ==================================================================
print("\n--- 8. Anthropic response_to_uir ---")

anthropic_response = {
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20241022",
    "content": [
        {"type": "text", "text": "Let me check the weather."},
        {"type": "tool_use", "id": "toolu_999", "name": "get_weather", "input": {"city": "Paris"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 150, "output_tokens": 42},
}

result8 = anthropic_adapter.response_to_uir(anthropic_response)
resp8 = result8.value
assert resp8.id == "msg_abc123"
assert resp8.stop_reason == "tool_calls"  # tool_use -> tool_calls
assert len(resp8.output[0].content) == 2
assert len(resp8.output[0].tool_calls) == 1
assert resp8.output[0].tool_calls[0].name == "get_weather"
assert resp8.usage.input_tokens == 150
print(f"{PASS} Anthropic response: stop={resp8.stop_reason}, tool={resp8.output[0].tool_calls[0].name}")

# ==================================================================
# Test 9: data URI base64 image parsing
# ==================================================================
print("\n--- 9. data URI base64 parsing ---")

openai_b64 = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ=="}},
            ],
        }
    ],
}
result9 = openai_adapter.request_to_uir(openai_b64)
img = result9.value.messages[0].content[1]
assert isinstance(img, ImageBlock)
assert img.source.kind == "base64"
assert img.source.mime_type == "image/jpeg"
assert img.source.value == "/9j/4AAQ=="
print(f"{PASS} data URI parsed: kind={img.source.kind}, mime={img.source.mime_type}")

print("\n" + "=" * 60)
print("All Step 4 validations passed!")
print("=" * 60)
