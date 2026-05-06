import asyncio
from pprint import pprint

from anyllm import AnthropicAdapter, OpenAIChatAdapter, UniversalConverter


async def main() -> None:
    converter = UniversalConverter()
    converter.register_adapter("openai_chat", OpenAIChatAdapter())
    converter.register_adapter("anthropic", AnthropicAdapter())

    openai_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "你是一个简洁的中文助手。"},
            {"role": "user", "content": "用两句话解释量子纠缠。"},
        ],
        "temperature": 0.4,
        "max_tokens": 256,
    }

    result = await converter.convert_request(
        source_provider="openai_chat",
        target_provider="anthropic",
        raw_request=openai_request,
    )

    print("=== Anthropic Request ===")
    pprint(result.value)

    print("\n=== Warnings ===")
    if not result.warnings:
        print("(none)")
    else:
        for warning in result.warnings:
            print(f"[{warning.severity}] {warning.code} @ {warning.path}: {warning.message}")


if __name__ == "__main__":
    asyncio.run(main())
