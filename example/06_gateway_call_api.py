import asyncio
import os

from anyllm import (
    AnthropicAdapter,
    AnyLLMGateway,
    GeminiAdapter,
    Message,
    ModelRef,
    ProviderConfig,
    UniversalRequest,
)


async def main() -> None:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not anthropic_api_key and not google_api_key:
        raise RuntimeError("请至少设置 ANTHROPIC_API_KEY 或 GOOGLE_API_KEY")

    gateway = AnyLLMGateway()

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

    if google_api_key:
        gateway.register_provider(
            "google",
            ProviderConfig(
                adapter=GeminiAdapter(),
                api_base="https://generativelanguage.googleapis.com",
                api_key=google_api_key,
                timeout=60.0,
            ),
        )

    if google_api_key:
        request = UniversalRequest(
            model=ModelRef(provider="google", name="gemini-1.5-flash"),
            messages=[Message.user_text("详细介绍 AnyLLM 的作用")],
        )
    else:
        request = UniversalRequest(
            model=ModelRef(provider="anthropic", name="claude-sonnet-4-5"),
            messages=[Message.user_text("详细介绍 AnyLLM 的作用")],
        )

    result = await gateway.chat_completions(request)

    print("=== Provider ===")
    print(result.provider)

    print("\n=== Assistant Output ===")
    for message in result.response.output:
        for block in message.content:
            if getattr(block, "type", None) == "text":
                print(block.text)

    print("\n=== Usage ===")
    if result.response.usage is None:
        print("(no usage returned)")
    else:
        print(result.response.usage.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
