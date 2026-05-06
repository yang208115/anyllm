import asyncio
from pprint import pprint

from anyllm import (
    AnthropicAdapter,
    OpenAIChatAdapter,
    TextBlock,
    UniversalConverter,
    interceptor,
)


@interceptor("append_disclaimer")
async def append_disclaimer(request, target_provider):
    if target_provider == "anthropic":
        request.instructions.append(TextBlock(text="请在回答结尾加一句：仅供参考。"))
    return request


async def main() -> None:
    converter = UniversalConverter()
    converter.register_adapter("openai_chat", OpenAIChatAdapter())
    converter.register_adapter("anthropic", AnthropicAdapter())
    converter.register_interceptor(append_disclaimer)

    openai_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "给我一条健康建议。"},
        ],
        "max_tokens": 128,
    }

    result = await converter.convert_request(
        source_provider="openai_chat",
        target_provider="anthropic",
        raw_request=openai_request,
    )

    print("=== Converted Request with Custom Interceptor ===")
    pprint(result.value)


if __name__ == "__main__":
    asyncio.run(main())
