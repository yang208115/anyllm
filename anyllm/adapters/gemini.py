"""
Google Gemini 适配器。

映射要点：
- instructions <-> systemInstruction.parts
- assistant <-> model, tool -> user(functionResponse)
- functionCall / functionResponse <-> ToolCall / ToolResult
- generationConfig 字段映射
"""

from __future__ import annotations

import uuid
from typing import Any

from anyllm.adapters.base import BaseAdapter, ProviderCapabilities
from anyllm.capabilities.matrix import GEMINI_CAPABILITIES
from anyllm.conversion.lowering import blocks_to_plain_text
from anyllm.schema.content import (
    AudioBlock,
    ContentBlock,
    FileBlock,
    ImageBlock,
    MediaSource,
    ProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResult,
    ToolResultBlock,
    parse_tool_arguments,
)
from anyllm.schema.message import Message
from anyllm.schema.request import GenerationConfig, ModelRef, UniversalRequest
from anyllm.schema.response import UniversalResponse, normalize_stop_reason
from anyllm.schema.tools import (
    AutoToolChoice,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    NoneToolChoice,
    RequiredToolChoice,
    ResponseFormat,
    SpecificToolChoice,
    TextResponseFormat,
    ToolChoice,
    ToolDef,
)
from anyllm.schema.usage import Usage
from anyllm.schema.warnings import ConversionResult, ConversionWarning

_KNOWN_REQUEST_FIELDS = {
    "model",
    "systemInstruction",
    "contents",
    "tools",
    "toolConfig",
    "generationConfig",
    "safetySettings",
}


class GeminiAdapter(BaseAdapter):
    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return GEMINI_CAPABILITIES

    def request_to_uir(
        self,
        raw_request: dict[str, Any],
    ) -> ConversionResult[UniversalRequest]:
        warnings: list[ConversionWarning] = []
        instructions: list[ContentBlock] = []
        messages: list[Message] = []

        system_instruction = raw_request.get("systemInstruction") or {}
        parts = system_instruction.get("parts") or []
        for i, part in enumerate(parts):
            instructions.extend(
                self._parts_to_blocks(part, warnings, f"systemInstruction.parts[{i}]", "google")
            )

        for idx, raw_msg in enumerate(raw_request.get("contents") or []):
            path = f"contents[{idx}]"
            role = self._map_gemini_role_to_uir(raw_msg.get("role", "user"))
            content: list[ContentBlock] = []
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResult] = []

            for j, part in enumerate(raw_msg.get("parts") or []):
                block_path = f"{path}.parts[{j}]"
                parsed_blocks = self._parts_to_blocks(part, warnings, block_path, "google")
                for block in parsed_blocks:
                    content.append(block)
                    if isinstance(block, ToolCallBlock):
                        tool_calls.append(block.call)
                    elif isinstance(block, ToolResultBlock):
                        tool_results.append(block.result)

            messages.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    provider={"raw": raw_msg},
                )
            )

        tools = self._parse_tools(raw_request.get("tools") or [], warnings)
        tool_choice = self._parse_tool_choice(raw_request.get("toolConfig"), warnings)

        gen = raw_request.get("generationConfig") or {}
        response_format = self._parse_response_format(gen)
        generation = GenerationConfig(
            temperature=gen.get("temperature"),
            top_p=gen.get("topP"),
            top_k=gen.get("topK"),
            max_output_tokens=gen.get("maxOutputTokens"),
            stop=gen.get("stopSequences"),
            seed=gen.get("seed"),
        )

        vendor = {
            k: v for k, v in raw_request.items() if k not in _KNOWN_REQUEST_FIELDS
        }

        uir = UniversalRequest(
            version="uai.v1",
            model=ModelRef(provider="google", name=raw_request.get("model", "")),
            instructions=instructions,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            generation=generation,
            stream=False,
            vendor={"google": vendor} if vendor else {},
        )
        return ConversionResult(value=uir, warnings=warnings)

    def response_to_uir(
        self,
        raw_response: dict[str, Any],
    ) -> ConversionResult[UniversalResponse]:
        warnings: list[ConversionWarning] = []
        output: list[Message] = []

        candidates = raw_response.get("candidates") or []
        if len(candidates) > 1:
            warnings.append(
                ConversionWarning(
                    code="RESPONSE_CANDIDATES_TRUNCATED",
                    path="candidates",
                    message="Gemini 返回了多个 candidates，当前仅保留第一个。",
                    severity="info",
                )
            )

        if candidates:
            first = candidates[0]
            content = first.get("content") or {}
            msg_role = self._map_gemini_role_to_uir(content.get("role", "model"))

            blocks: list[ContentBlock] = []
            calls: list[ToolCall] = []
            results: list[ToolResult] = []
            for i, part in enumerate(content.get("parts") or []):
                parsed = self._parts_to_blocks(part, warnings, f"candidates[0].content.parts[{i}]", "google")
                for block in parsed:
                    blocks.append(block)
                    if isinstance(block, ToolCallBlock):
                        calls.append(block.call)
                    elif isinstance(block, ToolResultBlock):
                        results.append(block.result)

            output.append(
                Message(
                    role=msg_role,
                    content=blocks,
                    tool_calls=calls,
                    tool_results=results,
                    provider={"raw": content},
                )
            )

        usage_meta = raw_response.get("usageMetadata") or {}
        usage = None
        if usage_meta:
            usage = Usage(
                input_tokens=usage_meta.get("promptTokenCount"),
                output_tokens=usage_meta.get("candidatesTokenCount"),
                total_tokens=usage_meta.get("totalTokenCount"),
                provider=usage_meta,
            )

        finish_reason = candidates[0].get("finishReason") if candidates else None
        model_name = raw_response.get("modelVersion") or raw_response.get("model")

        response = UniversalResponse(
            id=raw_response.get("responseId"),
            model=model_name,
            output=output,
            stop_reason=normalize_stop_reason("gemini", finish_reason),
            usage=usage,
            raw=raw_response,
        )
        return ConversionResult(value=response, warnings=warnings)

    def uir_to_request(
        self,
        request: UniversalRequest,
    ) -> ConversionResult[dict[str, Any]]:
        warnings: list[ConversionWarning] = []

        raw: dict[str, Any] = {
            "model": request.model.name,
            "contents": [],
        }

        if request.instructions:
            raw["systemInstruction"] = {
                "parts": self._blocks_to_parts(
                    request.instructions, warnings, "instructions", for_tool_message=False
                )
            }

        for idx, msg in enumerate(request.messages):
            path = f"messages[{idx}]"
            role = self._map_uir_role_to_gemini(msg.role)

            msg_parts = self._blocks_to_parts(msg.content, warnings, f"{path}.content", for_tool_message=(msg.role == "tool"))

            existing_call_ids = {
                block.call.id
                for block in msg.content
                if isinstance(block, ToolCallBlock)
            }
            for _tc_idx, call in enumerate(msg.tool_calls):
                if call.id in existing_call_ids:
                    continue
                msg_parts.append(
                    {
                        "functionCall": {
                            "id": call.id,
                            "name": call.name,
                            "args": call.arguments,
                        }
                    }
                )

            existing_result_ids = {
                block.result.call_id
                for block in msg.content
                if isinstance(block, ToolResultBlock)
            }
            for tr_idx, result in enumerate(msg.tool_results):
                if result.call_id in existing_result_ids:
                    continue
                msg_parts.append(
                    self._tool_result_to_part(
                        result,
                        warnings,
                        f"{path}.tool_results[{tr_idx}]",
                    )
                )

            if not msg_parts:
                msg_parts = [{"text": ""}]

            raw["contents"].append({"role": role, "parts": msg_parts})

        if request.tools:
            non_function_tools = [t for t in request.tools if t.type != "function"]
            for i, tool in enumerate(non_function_tools):
                warnings.append(
                    ConversionWarning(
                        code="BUILTIN_TOOL_NOT_SUPPORTED",
                        path=f"tools[{i}]",
                        message=f"Gemini generateContent 仅支持 function 工具，'{tool.type}' 已忽略。",
                    )
                )
            raw["tools"] = [self._tool_def_to_gemini(t) for t in request.tools if t.type == "function"]

        tool_config = self._tool_choice_to_gemini(request.tool_choice, warnings)
        if tool_config is not None:
            raw["toolConfig"] = tool_config

        gen = self._generation_to_gemini(request.generation)
        self._apply_response_format(gen, request.response_format, warnings)
        if gen:
            raw["generationConfig"] = gen

        if (
            request.state.conversation_id
            or request.state.previous_response_id
            or request.state.thread_id
            or request.state.run_id
            or request.state.provider_state
        ):
            warnings.append(
                ConversionWarning(
                    code="STATE_NOT_SUPPORTED",
                    path="state",
                    message="Gemini generateContent 不支持 UIR state 字段。",
                )
            )

        vendor_google = request.vendor.get("google", {})
        if vendor_google:
            raw.update(vendor_google)

        return ConversionResult(value=raw, warnings=warnings)

    def uir_to_response(
        self,
        response: UniversalResponse,
    ) -> ConversionResult[dict[str, Any]]:
        warnings: list[ConversionWarning] = []

        finish_reason_map = {
            "end_turn": "STOP",
            "max_tokens": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
            "stop_sequence": "STOP",
            "error": "SAFETY",
            "unknown": "STOP",
        }
        finish_reason = finish_reason_map.get(response.stop_reason, "STOP")

        msg = response.output[0] if response.output else Message(role="assistant", content=[])
        parts = self._blocks_to_parts(msg.content, warnings, "output[0].content", for_tool_message=(msg.role == "tool"))

        for call in msg.tool_calls:
            parts.append(
                {
                    "functionCall": {
                        "id": call.id,
                        "name": call.name,
                        "args": call.arguments,
                    }
                }
            )

        for i, result in enumerate(msg.tool_results):
            parts.append(self._tool_result_to_part(result, warnings, f"output[0].tool_results[{i}]"))

        raw: dict[str, Any] = {
            "responseId": response.id or "",
            "candidates": [
                {
                    "content": {
                        "role": self._map_uir_role_to_gemini(msg.role),
                        "parts": parts or [{"text": ""}],
                    },
                    "finishReason": finish_reason,
                }
            ],
        }

        if response.model:
            raw["modelVersion"] = response.model

        if response.usage:
            raw["usageMetadata"] = {
                "promptTokenCount": response.usage.input_tokens or 0,
                "candidatesTokenCount": response.usage.output_tokens or 0,
                "totalTokenCount": response.usage.total_tokens or 0,
            }

        return ConversionResult(value=raw, warnings=warnings)

    def _map_gemini_role_to_uir(self, role: str) -> str:
        if role == "model":
            return "assistant"
        if role in ("user", "assistant", "tool", "system", "developer"):
            return role
        return "user"

    def _map_uir_role_to_gemini(self, role: str) -> str:
        if role == "assistant":
            return "model"
        if role == "tool":
            return "user"
        if role in ("user", "model"):
            return role
        return "user"

    def _parts_to_blocks(
        self,
        part: dict[str, Any],
        warnings: list[ConversionWarning],
        path: str,
        provider: str,
    ) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []

        if "text" in part:
            blocks.append(TextBlock(text=part.get("text", "")))

        if "functionCall" in part:
            fn = part.get("functionCall") or {}
            raw_args = fn.get("args", {})
            args, raw, warn_code = parse_tool_arguments(raw_args)
            if warn_code:
                warnings.append(
                    ConversionWarning(
                        code=warn_code,
                        path=f"{path}.functionCall.args",
                        message="Gemini functionCall.args 不是合法 JSON。",
                    )
                )
            call_id = fn.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            call = ToolCall(
                id=call_id,
                name=fn.get("name", ""),
                arguments=args,
                raw_arguments=raw,
                provider={"raw": fn},
            )
            blocks.append(ToolCallBlock(call=call))

        if "functionResponse" in part:
            fr = part.get("functionResponse") or {}
            result_payload = fr.get("response")
            is_error = False
            if isinstance(result_payload, str):
                result_content = [TextBlock(text=result_payload)]
            elif isinstance(result_payload, dict):
                is_error = bool(result_payload.get("is_error", False))
                output = result_payload.get("output", "")
                result_content = [TextBlock(text=str(output if output is not None else ""))]
            else:
                result_content = [TextBlock(text=str(result_payload if result_payload is not None else ""))]
            call_id = fr.get("id") or fr.get("call_id") or fr.get("name", "")
            result = ToolResult(
                call_id=call_id,
                name=fr.get("name"),
                content=result_content,
                is_error=is_error,
                provider={"raw": fr},
            )
            blocks.append(ToolResultBlock(result=result))

        if "inlineData" in part:
            inline = part.get("inlineData") or {}
            mime_type = inline.get("mimeType")
            data = inline.get("data", "")
            if mime_type and mime_type.startswith("image/"):
                blocks.append(
                    ImageBlock(
                        source=MediaSource(kind="base64", value=data, mime_type=mime_type)
                    )
                )
            elif mime_type and mime_type.startswith("audio/"):
                blocks.append(
                    AudioBlock(
                        source=MediaSource(kind="base64", value=data, mime_type=mime_type),
                        format=mime_type.split("/")[-1],
                    )
                )
            else:
                blocks.append(ProviderBlock(provider=provider, value=part))
                warnings.append(
                    ConversionWarning(
                        code="VENDOR_FIELD_PASSTHROUGH",
                        path=f"{path}.inlineData",
                        message="未识别的 inlineData 类型，已透传。",
                        severity="info",
                    )
                )

        if "fileData" in part:
            fd = part.get("fileData") or {}
            blocks.append(
                FileBlock(
                    source=MediaSource(
                        kind="file_id",
                        value=fd.get("fileUri") or fd.get("fileId") or "",
                        mime_type=fd.get("mimeType"),
                    ),
                    mime_type=fd.get("mimeType"),
                )
            )

        known_keys = {"text", "functionCall", "functionResponse", "inlineData", "fileData"}
        unknown = {k: v for k, v in part.items() if k not in known_keys}
        if unknown and not blocks:
            blocks.append(ProviderBlock(provider=provider, value=part))
            warnings.append(
                ConversionWarning(
                    code="VENDOR_FIELD_PASSTHROUGH",
                    path=path,
                    message="未知 Gemini part 已透传为 ProviderBlock。",
                    severity="info",
                )
            )

        return blocks

    def _blocks_to_parts(
        self,
        blocks: list[ContentBlock],
        warnings: list[ConversionWarning],
        path: str,
        *,
        for_tool_message: bool,
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []

        for i, block in enumerate(blocks):
            block_path = f"{path}[{i}]"

            if isinstance(block, TextBlock):
                parts.append({"text": block.text})
                continue

            if isinstance(block, ImageBlock):
                if block.source.kind == "base64":
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": block.source.mime_type or "image/png",
                                "data": block.source.value,
                            }
                        }
                    )
                elif block.source.kind == "url":
                    parts.append(
                        {
                            "fileData": {
                                "mimeType": block.source.mime_type or "image/png",
                                "fileUri": block.source.value,
                            }
                        }
                    )
                else:
                    warnings.append(
                        ConversionWarning(
                            code="UNSUPPORTED_MODALITY",
                            path=block_path,
                            message=f"Gemini image source '{block.source.kind}' 不支持，已降级为文本。",
                        )
                    )
                    parts.append({"text": "[Image omitted]"})
                continue

            if isinstance(block, AudioBlock):
                if block.source.kind == "base64":
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": block.source.mime_type or f"audio/{block.format or 'wav'}",
                                "data": block.source.value,
                            }
                        }
                    )
                else:
                    warnings.append(
                        ConversionWarning(
                            code="UNSUPPORTED_MODALITY",
                            path=block_path,
                            message=f"Gemini audio source '{block.source.kind}' 不支持，已降级为文本。",
                        )
                    )
                    parts.append({"text": "[Audio omitted]"})
                continue

            if isinstance(block, FileBlock):
                if block.source.kind in ("url", "file_id"):
                    parts.append(
                        {
                            "fileData": {
                                "mimeType": block.mime_type or block.source.mime_type or "application/octet-stream",
                                "fileUri": block.source.value,
                            }
                        }
                    )
                else:
                    warnings.append(
                        ConversionWarning(
                            code="FILE_REFERENCE_NOT_SUPPORTED",
                            path=block_path,
                            message=f"Gemini file source '{block.source.kind}' 不支持，已降级为文本。",
                        )
                    )
                    parts.append({"text": "[File omitted]"})
                continue

            if isinstance(block, ToolCallBlock):
                parts.append(
                    {
                        "functionCall": {
                            "id": block.call.id,
                            "name": block.call.name,
                            "args": block.call.arguments,
                        }
                    }
                )
                continue

            if isinstance(block, ToolResultBlock):
                parts.append(self._tool_result_to_part(block.result, warnings, block_path))
                continue

            if isinstance(block, RefusalBlock):
                parts.append({"text": block.text})
                continue

            if isinstance(block, ThinkingBlock):
                if block.text:
                    parts.append({"text": f"[Thinking: {block.text}]"})
                    warnings.append(
                        ConversionWarning(
                            code="UNSUPPORTED_CONTENT_BLOCK",
                            path=block_path,
                            message="Gemini 不直接支持 ThinkingBlock，已降级为文本。",
                            severity="info",
                        )
                    )
                continue

            if isinstance(block, ProviderBlock):
                if block.provider == "google" and isinstance(block.value, dict):
                    parts.append(block.value)
                else:
                    warnings.append(
                        ConversionWarning(
                            code="UNSUPPORTED_CONTENT_BLOCK",
                            path=block_path,
                            message=f"来自 '{block.provider}' 的私有块无法直接映射到 Gemini。",
                        )
                    )
                    parts.append({"text": "[Unsupported content]"})
                continue

            warnings.append(
                ConversionWarning(
                    code="UNSUPPORTED_CONTENT_BLOCK",
                    path=block_path,
                    message=f"不支持的内容块: {type(block).__name__}，已降级为文本。",
                )
            )
            parts.append({"text": "[Unsupported content]"})

        return parts

    def _tool_result_to_part(
        self,
        result: ToolResult,
        warnings: list[ConversionWarning],
        path: str,
    ) -> dict[str, Any]:
        return {
            "functionResponse": {
                "id": result.call_id,
                "name": result.name or "tool",
                "response": {
                    "output": blocks_to_plain_text(result.content, warnings, f"{path}.content"),
                    "is_error": result.is_error,
                },
            }
        }

    def _parse_tools(
        self,
        tools: list[dict[str, Any]],
        warnings: list[ConversionWarning],
    ) -> list[ToolDef]:
        out: list[ToolDef] = []
        for _i, t in enumerate(tools):
            declarations = t.get("functionDeclarations") or []
            for _j, fn in enumerate(declarations):
                out.append(
                    ToolDef(
                        type="function",
                        name=fn.get("name", ""),
                        description=fn.get("description"),
                        input_schema=fn.get("parameters"),
                        provider={"raw": fn},
                    )
                )
        return out

    def _tool_def_to_gemini(self, tool: ToolDef) -> dict[str, Any]:
        fn: dict[str, Any] = {
            "name": tool.name,
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        }
        if tool.description:
            fn["description"] = tool.description
        return {"functionDeclarations": [fn]}

    def _parse_tool_choice(
        self,
        raw_tool_config: Any,
        warnings: list[ConversionWarning],
    ) -> ToolChoice | None:
        if raw_tool_config is None:
            return None

        function_calling = raw_tool_config.get("functionCallingConfig") or {}
        mode = function_calling.get("mode")
        allowed = function_calling.get("allowedFunctionNames") or []

        if mode == "AUTO":
            return AutoToolChoice()
        if mode == "NONE":
            return NoneToolChoice()
        if mode == "ANY":
            if len(allowed) == 1:
                return SpecificToolChoice(name=allowed[0])
            return RequiredToolChoice()

        warnings.append(
            ConversionWarning(
                code="TOOL_CHOICE_DOWNGRADED",
                path="toolConfig.functionCallingConfig.mode",
                message="未知 Gemini tool choice mode，已降级为 AUTO。",
                severity="info",
            )
        )
        return AutoToolChoice()

    def _tool_choice_to_gemini(
        self,
        choice: ToolChoice | None,
        warnings: list[ConversionWarning],
    ) -> dict[str, Any] | None:
        if choice is None:
            return None

        if isinstance(choice, AutoToolChoice):
            return {"functionCallingConfig": {"mode": "AUTO"}}

        if isinstance(choice, NoneToolChoice):
            return {"functionCallingConfig": {"mode": "NONE"}}

        if isinstance(choice, RequiredToolChoice):
            return {"functionCallingConfig": {"mode": "ANY"}}

        if isinstance(choice, SpecificToolChoice):
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [choice.name],
                }
            }

        warnings.append(
            ConversionWarning(
                code="TOOL_CHOICE_DOWNGRADED",
                path="tool_choice",
                message="未知 tool_choice，已降级为 AUTO。",
                severity="info",
            )
        )
        return {"functionCallingConfig": {"mode": "AUTO"}}

    def _parse_response_format(
        self,
        generation_config: dict[str, Any],
    ) -> ResponseFormat | None:
        mime = generation_config.get("responseMimeType")
        schema = generation_config.get("responseSchema")
        if mime == "application/json" and schema:
            return JsonSchemaResponseFormat(name="gemini_schema", schema=schema, strict=False)
        if mime == "application/json":
            return JsonObjectResponseFormat()
        return None

    def _apply_response_format(
        self,
        out_generation_config: dict[str, Any],
        response_format: ResponseFormat | None,
        warnings: list[ConversionWarning],
    ) -> None:
        if response_format is None:
            return

        if isinstance(response_format, TextResponseFormat):
            return

        if isinstance(response_format, JsonObjectResponseFormat):
            out_generation_config["responseMimeType"] = "application/json"
            return

        if isinstance(response_format, JsonSchemaResponseFormat):
            out_generation_config["responseMimeType"] = "application/json"
            out_generation_config["responseSchema"] = response_format.json_schema
            return

        warnings.append(
            ConversionWarning(
                code="RESPONSE_FORMAT_DOWNGRADED",
                path="response_format",
                message="Gemini 未识别的 response_format，已降级为默认文本输出。",
            )
        )

    def _generation_to_gemini(self, generation: GenerationConfig) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if generation.temperature is not None:
            out["temperature"] = generation.temperature
        if generation.top_p is not None:
            out["topP"] = generation.top_p
        if generation.top_k is not None:
            out["topK"] = generation.top_k
        if generation.max_output_tokens is not None:
            out["maxOutputTokens"] = generation.max_output_tokens
        if generation.stop is not None:
            out["stopSequences"] = generation.stop
        if generation.seed is not None:
            out["seed"] = generation.seed
        return out
