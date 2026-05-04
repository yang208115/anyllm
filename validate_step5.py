# -*- coding: utf-8 -*-
"""Step 5 validation - AnyLLMGateway full integration test."""
import asyncio
import json

from anyllm import (
    AnyLLMGateway,
    ProviderConfig,
    OpenAIChatAdapter,
    AnthropicAdapter,
    ImageResolutionInterceptor,
    RoleConsolidationInterceptor,
    FunctionInterceptor,
    interceptor,
    UniversalRequest,
    ModelRef,
    Message,
    TextBlock,
    ImageBlock,
    MediaSource,
    ToolCall,
    ToolResult,
    ToolDef,
    AutoToolChoice,
    GenerationConfig,
)

PASS = "[OK]"

print("=" * 60)
print("Step 5: AnyLLMGateway Integration Test")
print("=" * 60)

# ==================================================================
# 1. Gateway initialization & registration
# ==================================================================
print("\n--- 1. Gateway registration ---")

gateway = AnyLLMGateway()

gateway.register_provider("openai_chat", ProviderConfig(
    adapter=OpenAIChatAdapter(),
    api_base="https://api.openai.com/v1",
    api_key="sk-test-key",
))
gateway.register_provider("anthropic", ProviderConfig(
    adapter=AnthropicAdapter(),
    api_base="https://api.anthropic.com",
    api_key="sk-ant-test-key",
))

gateway.register_interceptor(ImageResolutionInterceptor())
gateway.register_interceptor(RoleConsolidationInterceptor())

assert gateway.registered_providers == ["openai_chat", "anthropic"]
assert gateway.registered_interceptors == ["image_resolution", "role_consolidation"]
print(f"{PASS} Providers: {gateway.registered_providers}")
print(f"{PASS} Interceptors: {gateway.registered_interceptors}")

# ==================================================================
# 2. Provider routing
# ==================================================================
print("\n--- 2. Provider routing ---")

# Route by model.provider
req_openai = UniversalRequest(
    model=ModelRef(provider="openai", name="gpt-4o"),
    messages=[Message.user_text("Hi")],
)
provider = gateway._resolve_provider(req_openai)
assert provider == "openai_chat"
print(f"{PASS} ModelRef(provider='openai') -> '{provider}'")

req_anthropic = UniversalRequest(
    model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
    messages=[Message.user_text("Hi")],
)
provider2 = gateway._resolve_provider(req_anthropic)
assert provider2 == "anthropic"
print(f"{PASS} ModelRef(provider='anthropic') -> '{provider2}'")

# Route by model name inference
req_name = UniversalRequest(
    model=ModelRef(name="claude-sonnet-4-5"),
    messages=[Message.user_text("Hi")],
)
provider3 = gateway._resolve_provider(req_name)
assert provider3 == "anthropic"
print(f"{PASS} Model name 'claude-sonnet-4-5' -> '{provider3}'")

req_gpt = UniversalRequest(
    model=ModelRef(name="gpt-4o-mini"),
    messages=[Message.user_text("Hi")],
)
provider4 = gateway._resolve_provider(req_gpt)
assert provider4 == "openai_chat"
print(f"{PASS} Model name 'gpt-4o-mini' -> '{provider4}'")

# ==================================================================
# 3. Custom router
# ==================================================================
print("\n--- 3. Custom router ---")

def my_router(request):
    if "fast" in request.model.name:
        return "openai_chat"
    return "anthropic"

gateway.set_router(my_router)

req_fast = UniversalRequest(
    model=ModelRef(name="fast-model"),
    messages=[Message.user_text("Hi")],
)
assert gateway._resolve_provider(req_fast) == "openai_chat"

req_slow = UniversalRequest(
    model=ModelRef(name="slow-model"),
    messages=[Message.user_text("Hi")],
)
assert gateway._resolve_provider(req_slow) == "anthropic"
print(f"{PASS} Custom router works: fast->openai_chat, slow->anthropic")

# Reset router
gateway.set_router(None)

# ==================================================================
# 4. convert_only (no API call)
# ==================================================================
print("\n--- 4. convert_only ---")

async def test_convert_only():
    # Build a complex request with tool calling
    request = UniversalRequest(
        model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
        instructions=[TextBlock(text="You are a weather assistant.")],
        messages=[
            Message.user_text("Tokyo weather?"),
            Message(
                role="assistant",
                content=[TextBlock(text="Let me check.")],
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
                ],
            ),
            Message(
                role="tool",
                content=[],
                tool_results=[
                    ToolResult(
                        call_id="call_1",
                        name="get_weather",
                        content=[TextBlock(text="Sunny, 25C")],
                    )
                ],
            ),
            Message.user_text("Thanks, now what about Paris?"),
        ],
        tools=[ToolDef(
            name="get_weather",
            description="Get weather by city",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )],
        tool_choice=AutoToolChoice(),
        generation=GenerationConfig(temperature=0.5, max_output_tokens=2048),
    )

    # Convert to Anthropic format
    result = await gateway.convert_only(request, target_provider="anthropic")
    out = result.value

    # Verify Anthropic format
    assert out["model"] == "claude-sonnet-4-5"
    assert out["max_tokens"] == 2048
    assert out["system"] == "You are a weather assistant."
    assert out["temperature"] == 0.5
    print(f"  {PASS} Model: {out['model']}, max_tokens: {out['max_tokens']}")

    # Messages should not contain system role (hoisted by RoleConsolidationInterceptor)
    msg_roles = [m["role"] for m in out["messages"]]
    assert "system" not in msg_roles
    print(f"  {PASS} Message roles: {msg_roles}")

    # After interceptor: tool messages should be merged into user
    # Original: [user, assistant, tool, user]
    # After consolidation: [user, assistant, user(tool_result), user] -> merged -> [user, assistant, user]
    assert all(r in ("user", "assistant") for r in msg_roles)
    print(f"  {PASS} No 'tool' role in output (merged by interceptor)")

    # Check tool_use in assistant message
    asst = out["messages"][1]
    tool_use_blocks = [b for b in asst["content"] if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "get_weather"
    print(f"  {PASS} tool_use block in assistant: {tool_use_blocks[0]['name']}")

    # Check tool_result in user message
    has_tool_result = False
    for msg in out["messages"]:
        for b in msg.get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                has_tool_result = True
                assert b["tool_use_id"] == "call_1"
    assert has_tool_result
    print(f"  {PASS} tool_result block in user message")

    # Check tools definition
    assert len(out["tools"]) == 1
    assert out["tools"][0]["name"] == "get_weather"
    print(f"  {PASS} Tools: {[t['name'] for t in out['tools']]}")

    # Check warnings
    print(f"  {PASS} Warnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"    [{w.severity}] {w.code}")

    # --- Also convert to OpenAI format ---
    # IMPORTANT: We need a fresh request because the interceptor pipeline
    # mutates the request in-place (RoleConsolidationInterceptor merged tool→user).
    request_oai = UniversalRequest(
        model=ModelRef(provider="openai", name="gpt-4o"),
        instructions=[TextBlock(text="You are a weather assistant.")],
        messages=[
            Message.user_text("Tokyo weather?"),
            Message(
                role="assistant",
                content=[TextBlock(text="Let me check.")],
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
                ],
            ),
            Message(
                role="tool",
                content=[],
                tool_results=[
                    ToolResult(call_id="call_1", name="get_weather",
                               content=[TextBlock(text="Sunny, 25C")])
                ],
            ),
            Message.user_text("Thanks"),
        ],
    )
    # RoleConsolidationInterceptor does NOT run for openai_chat
    # (not in strict-alternation set), so messages keep original structure.
    result_oai = await gateway.convert_only(request_oai, target_provider="openai_chat")
    out_oai = result_oai.value

    assert out_oai["model"] == "gpt-4o"
    assert out_oai["messages"][0]["role"] == "system"
    assert out_oai["messages"][0]["content"] == "You are a weather assistant."

    # Find assistant message with tool_calls
    asst_msgs = [m for m in out_oai["messages"] if m["role"] == "assistant"]
    assert len(asst_msgs) >= 1
    asst_with_tools = [m for m in asst_msgs if "tool_calls" in m]
    assert len(asst_with_tools) == 1
    assert asst_with_tools[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    # Find tool role message
    tool_msgs = [m for m in out_oai["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    print(f"  {PASS} Also verified OpenAI format conversion")

asyncio.run(test_convert_only())

# ==================================================================
# 5. Custom interceptor with gateway
# ==================================================================
print("\n--- 5. Custom interceptor with gateway ---")

async def test_custom_interceptor():
    gw = AnyLLMGateway()
    gw.register_provider("openai_chat", ProviderConfig(
        adapter=OpenAIChatAdapter(),
    ))

    # Register custom interceptor using decorator
    @interceptor("add_tag", only_for={"openai_chat"})
    async def add_tag(request, target_provider):
        request.instructions.append(TextBlock(text="[AnyLLM Gateway v0.1]"))
        return request

    gw.register_interceptor(ImageResolutionInterceptor())
    gw.register_interceptor(add_tag)
    gw.register_interceptor(RoleConsolidationInterceptor())

    assert gw.registered_interceptors == [
        "image_resolution", "add_tag", "role_consolidation"
    ]
    print(f"  {PASS} Interceptor order: {gw.registered_interceptors}")

    req = UniversalRequest(
        model=ModelRef(provider="openai", name="gpt-4o"),
        instructions=[TextBlock(text="Be helpful.")],
        messages=[Message.user_text("Hi")],
    )
    result = await gw.convert_only(req, target_provider="openai_chat")
    out = result.value

    # System message should contain both instructions
    system_msg = out["messages"][0]
    assert "[AnyLLM Gateway v0.1]" in system_msg["content"]
    assert "Be helpful." in system_msg["content"]
    print(f"  {PASS} Custom interceptor added tag to system prompt")

asyncio.run(test_custom_interceptor())

# ==================================================================
# 6. Full import test
# ==================================================================
print("\n--- 6. Full import test ---")
import anyllm
assert hasattr(anyllm, 'AnyLLMGateway')
assert hasattr(anyllm, 'UniversalConverter')
assert hasattr(anyllm, 'OpenAIChatAdapter')
assert hasattr(anyllm, 'AnthropicAdapter')
assert hasattr(anyllm, 'ImageResolutionInterceptor')
assert hasattr(anyllm, 'RoleConsolidationInterceptor')
assert hasattr(anyllm, 'FunctionInterceptor')
assert hasattr(anyllm, 'interceptor')
assert anyllm.__version__ == "0.1.0"
print(f"{PASS} All top-level exports accessible, version={anyllm.__version__}")

print("\n" + "=" * 60)
print("All Step 5 validations passed!")
print("=" * 60)
