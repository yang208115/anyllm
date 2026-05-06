from pprint import pprint

from anyllm import (
    AutoToolChoice,
    GenerationConfig,
    Message,
    ModelRef,
    ToolDef,
    UniversalRequest,
)


def main() -> None:
    request = UniversalRequest(
        model=ModelRef(provider="openai", name="gpt-4o"),
        instructions=[],
        messages=[
            Message.user_text("帮我查一下上海今天的天气。"),
        ],
        tools=[
            ToolDef(
                name="get_weather",
                description="获取指定城市天气",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        ],
        tool_choice=AutoToolChoice(),
        generation=GenerationConfig(temperature=0.3, max_output_tokens=256),
    )

    print("=== UniversalRequest (UIR) ===")
    pprint(request.model_dump(mode="json", by_alias=True))


if __name__ == "__main__":
    main()
