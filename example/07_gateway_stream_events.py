import asyncio
import os

from anyllm import (
    AnthropicAdapter,
    AnyLLMGateway,
    Message,
    ModelRef,
    OpenAIChatAdapter,
    ProviderConfig,
    UniversalRequest,
)


async def main() -> None:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_api_key and not anthropic_api_key:
        raise RuntimeError("请设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY")

    gateway = AnyLLMGateway()

    if openai_api_key:
        gateway.register_provider(
            "openai_chat",
            ProviderConfig(
                adapter=OpenAIChatAdapter(),
                api_base="https://api.openai.com/v1",
                api_key=openai_api_key,
                timeout=60.0,
            ),
        )

    if anthropic_api_key:
        gateway.register_provider(
            "anthropic",
            ProviderConfig(
                adapter=AnthropicAdapter(),
                api_base="https://api.anthropic.com",
                api_key=anthropic_api_key,
                timeout=60.0,
            ),
        )

    if openai_api_key:
        provider = "openai_chat"
        model = ModelRef(provider="openai", name="gpt-4o-mini")
    else:
        provider = "anthropic"
        model = ModelRef(provider="anthropic", name="claude-sonnet-4-5")

    request = UniversalRequest(
        model=model,
        messages=[Message.user_text("请用三点说明 AnyLLM 的核心价值。")],
        stream=True,
    )

    print(f"=== Provider: {provider} ===")
    print("=== Streaming Output ===")

    async for event in gateway.chat_completions_stream(request, provider=provider):
        if event.type == "content_delta" and event.delta is not None and getattr(event.delta, "type", None) == "text":
            print(event.delta.text, end="", flush=True)
        elif event.type == "tool_call_started" and event.tool_call is not None:
            print(f"\n[tool_call_started] {event.tool_call.name} ({event.tool_call.id})")
        elif event.type == "error":
            print("\n[stream_error]", event.raw)

    print("\n\n=== Stream Completed ===")


if __name__ == "__main__":
    asyncio.run(main())
