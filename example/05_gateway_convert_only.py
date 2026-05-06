import asyncio
from pprint import pprint

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
    gateway = AnyLLMGateway()
    gateway.register_provider(
        "openai_chat",
        ProviderConfig(adapter=OpenAIChatAdapter(), api_base="https://api.openai.com/v1"),
    )
    gateway.register_provider(
        "anthropic",
        ProviderConfig(adapter=AnthropicAdapter(), api_base="https://api.anthropic.com"),
    )

    request = UniversalRequest(
        model=ModelRef(provider="anthropic", name="claude-sonnet-4-6"),
        messages=[Message.user_text("给我三个 Python 学习建议")],
    )

    result = await gateway.convert_only(request, target_provider="anthropic")

    print("=== Gateway convert_only result ===")
    pprint(result.value)

    print("\n=== Warnings ===")
    if not result.warnings:
        print("(none)")
    else:
        for warning in result.warnings:
            print(f"[{warning.severity}] {warning.code} @ {warning.path}: {warning.message}")


if __name__ == "__main__":
    asyncio.run(main())
