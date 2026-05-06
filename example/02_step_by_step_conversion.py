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
            {"role": "system", "content": "你是代码审阅助手。"},
            {"role": "user", "content": "请审阅这段函数命名是否清晰。"},
        ],
        "temperature": 0.8,
        "max_tokens": 200,
    }

    uir_result = converter.request_to_uir("openai_chat", openai_request)
    uir_request = uir_result.value

    uir_request.generation.temperature = 0.2

    processed = await converter.run_interceptors(uir_request, "anthropic")
    target_result = converter.uir_to_request("anthropic", processed)

    warnings = uir_result.warnings + target_result.warnings

    print("=== UIR Model ===")
    print(uir_request.model_dump_json(indent=2, ensure_ascii=False))

    print("\n=== Anthropic Request ===")
    pprint(target_result.value)

    print("\n=== Warnings ===")
    if not warnings:
        print("(none)")
    else:
        for warning in warnings:
            print(f"[{warning.severity}] {warning.code} @ {warning.path}: {warning.message}")


if __name__ == "__main__":
    asyncio.run(main())
