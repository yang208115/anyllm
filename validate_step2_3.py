"""Step 2 & 3 validation script."""
import asyncio

from anyllm.capabilities.matrix import (
    ANTHROPIC_CAPABILITIES,
    BEDROCK_CONVERSE_CAPABILITIES,
    GEMINI_CAPABILITIES,
    OLLAMA_OPENAI_COMPAT_CAPABILITIES,
    OPENAI_CHAT_CAPABILITIES,
)
from anyllm.conversion.converter import UniversalConverter
from anyllm.conversion.lowering import (
    blocks_to_plain_text,
    extract_text_from_blocks,
    serialize_tool_arguments,
)
from anyllm.interceptors import (
    ImageResolutionInterceptor,
    RoleConsolidationInterceptor,
)
from anyllm.schema import (
    ImageBlock,
    MediaSource,
    Message,
    ModelRef,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    UniversalRequest,
)

PASS = "[OK]"

print("=" * 60)
print("Step 2 & 3: Interfaces + Interceptors Validation")
print("=" * 60)

# ==================================================================
# 1. blocks_to_plain_text
# ==================================================================
print("\n--- 1. blocks_to_plain_text ---")
warnings = []
text = blocks_to_plain_text(
    [
        TextBlock(text="Hello"),
        ImageBlock(source=MediaSource(kind="url", value="https://img.png")),
        TextBlock(text="World"),
    ],
    warnings,
    "test.content",
)
assert text == "Hello\n[Image omitted]\nWorld"
assert len(warnings) == 1
assert warnings[0].code == "UNSUPPORTED_MODALITY"
print(f"{PASS} blocks_to_plain_text: '{text}' (warnings: {len(warnings)})")

# ==================================================================
# 2. extract_text_from_blocks
# ==================================================================
print("\n--- 2. extract_text_from_blocks ---")
plain = extract_text_from_blocks([
    TextBlock(text="Line 1"),
    ImageBlock(source=MediaSource(kind="url", value="img")),
    TextBlock(text="Line 2"),
])
assert plain == "Line 1\nLine 2"
print(f"{PASS} extract_text_from_blocks: '{plain}'")

# ==================================================================
# 3. serialize_tool_arguments
# ==================================================================
print("\n--- 3. serialize_tool_arguments ---")
assert serialize_tool_arguments({"city": "Tokyo"}) == '{"city": "Tokyo"}'
assert serialize_tool_arguments("raw string") == "raw string"
print(f"{PASS} serialize_tool_arguments")

# ==================================================================
# 4. ProviderCapabilities
# ==================================================================
print("\n--- 4. ProviderCapabilities ---")
assert OPENAI_CHAT_CAPABILITIES.image_input is True
assert OPENAI_CHAT_CAPABILITIES.stateful is False
assert ANTHROPIC_CAPABILITIES.tool_result_blocks is True
assert ANTHROPIC_CAPABILITIES.json_schema is False
assert GEMINI_CAPABILITIES.audio_input is True
assert BEDROCK_CONVERSE_CAPABILITIES.file_input is False
assert OLLAMA_OPENAI_COMPAT_CAPABILITIES.tools is True
print(f"{PASS} Provider capabilities matrix verified (5 providers)")

# ==================================================================
# 5. UniversalConverter registration
# ==================================================================
print("\n--- 5. UniversalConverter ---")
converter = UniversalConverter()

# Should raise KeyError for unregistered provider
try:
    converter.get_adapter("nonexistent")
    raise AssertionError("Should have raised KeyError")
except KeyError:
    print(f"{PASS} get_adapter raises KeyError for unknown provider")

# Verify interceptor registration
converter.register_interceptor(ImageResolutionInterceptor())
converter.register_interceptor(RoleConsolidationInterceptor())
print(f"{PASS} Registered 2 interceptors")

# ==================================================================
# 6. RoleConsolidationInterceptor
# ==================================================================
print("\n--- 6. RoleConsolidationInterceptor ---")

async def test_role_consolidation():
    interceptor = RoleConsolidationInterceptor()

    # --- 6a: system message hoisting ---
    req = UniversalRequest(
        model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
        messages=[
            Message(role="system", content=[TextBlock(text="You are helpful.")]),
            Message(role="user", content=[TextBlock(text="Hello")]),
            Message(role="assistant", content=[TextBlock(text="Hi!")]),
        ],
    )
    req = await interceptor.process(req, target_provider="anthropic")
    # system message should be hoisted to instructions
    assert len(req.instructions) == 1
    assert all(m.role != "system" for m in req.messages)
    print(f"  {PASS} 6a: system message hoisted to instructions")

    # --- 6b: tool messages merged into user ---
    req2 = UniversalRequest(
        model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
        messages=[
            Message(role="user", content=[TextBlock(text="Query weather")]),
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(id="c1", name="get_weather", arguments={"city": "Tokyo"}),
                    ToolCall(id="c2", name="get_time", arguments={"tz": "JST"}),
                ],
            ),
            # Two consecutive tool messages (parallel tool results)
            Message(
                role="tool",
                content=[],
                tool_results=[
                    ToolResult(call_id="c1", content=[TextBlock(text="Sunny")])
                ],
            ),
            Message(
                role="tool",
                content=[],
                tool_results=[
                    ToolResult(call_id="c2", content=[TextBlock(text="14:00 JST")])
                ],
            ),
            Message(role="user", content=[TextBlock(text="Thanks")]),
        ],
    )
    req2 = await interceptor.process(req2, target_provider="anthropic")
    # tool messages should be merged into a user message
    assert all(m.role != "tool" for m in req2.messages), (
        f"Expected no tool messages, got roles: {[m.role for m in req2.messages]}"
    )
    # Check that tool results became content blocks in a user message
    tool_result_blocks = []
    for msg in req2.messages:
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                tool_result_blocks.append(block)
    assert len(tool_result_blocks) == 2, (
        f"Expected 2 ToolResultBlocks, got {len(tool_result_blocks)}"
    )
    print(f"  {PASS} 6b: tool messages merged into user ({len(tool_result_blocks)} ToolResultBlocks)")

    # --- 6c: consecutive same-role merge ---
    req3 = UniversalRequest(
        model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
        messages=[
            Message(role="user", content=[TextBlock(text="A")]),
            Message(role="user", content=[TextBlock(text="B")]),
            Message(role="assistant", content=[TextBlock(text="C")]),
            Message(role="assistant", content=[TextBlock(text="D")]),
            Message(role="user", content=[TextBlock(text="E")]),
        ],
    )
    req3 = await interceptor.process(req3, target_provider="anthropic")
    roles = [m.role for m in req3.messages]
    assert roles == ["user", "assistant", "user"], (
        f"Expected [user, assistant, user], got {roles}"
    )
    # First user message should have 2 text blocks
    assert len(req3.messages[0].content) == 2
    print(f"  {PASS} 6c: consecutive same-role merged -> {roles}")

    # --- 6d: no-op for non-strict providers ---
    req4 = UniversalRequest(
        model=ModelRef(provider="openai", name="gpt-4o"),
        messages=[
            Message(role="user", content=[TextBlock(text="A")]),
            Message(role="user", content=[TextBlock(text="B")]),
        ],
    )
    req4 = await interceptor.process(req4, target_provider="openai_chat")
    # Should NOT merge for OpenAI
    assert len(req4.messages) == 2
    print(f"  {PASS} 6d: no-op for non-strict provider (openai_chat)")


asyncio.run(test_role_consolidation())
print(f"{PASS} RoleConsolidationInterceptor all checks passed")

# ==================================================================
# 7. ImageResolutionInterceptor (without network)
# ==================================================================
print("\n--- 7. ImageResolutionInterceptor ---")

async def test_image_resolution():
    interceptor = ImageResolutionInterceptor()

    # Already base64 -> should not change
    req = UniversalRequest(
        model=ModelRef(provider="anthropic", name="test"),
        messages=[
            Message(role="user", content=[
                ImageBlock(source=MediaSource(kind="base64", value="abc", mime_type="image/png")),
            ]),
        ],
    )
    req = await interceptor.process(req, target_provider="anthropic")
    img = req.messages[0].content[0]
    assert isinstance(img, ImageBlock)
    assert img.source.kind == "base64"
    print(f"  {PASS} 7a: base64 image unchanged for anthropic")

    # URL image for openai_chat -> should skip (openai supports URL)
    req2 = UniversalRequest(
        model=ModelRef(provider="openai", name="gpt-4o"),
        messages=[
            Message(role="user", content=[
                ImageBlock(source=MediaSource(kind="url", value="https://example.com/img.png")),
            ]),
        ],
    )
    req2 = await interceptor.process(req2, target_provider="openai_chat")
    img2 = req2.messages[0].content[0]
    assert isinstance(img2, ImageBlock)
    assert img2.source.kind == "url"  # unchanged
    print(f"  {PASS} 7b: URL image unchanged for openai_chat (supports URL)")


asyncio.run(test_image_resolution())
print(f"{PASS} ImageResolutionInterceptor basic checks passed")

print("\n" + "=" * 60)
print("All Step 2 & 3 validations passed!")
print("=" * 60)
