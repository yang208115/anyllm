"""
Usage — 统一 Token 用量结构（PRD §25）。

设计原则：
  不强行统一所有计费字段，provider 原始用量保留在 provider 字段中。
  只抽取通用的 input/output/total token 数以及 reasoning / cache 加速相关指标。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class Usage(BaseModel):
    """统一 token 用量，各 provider 的原始用量保留在 provider 字段。"""

    model_config = ConfigDict(populate_by_name=True)

    input_tokens: Optional[int] = None
    """输入（prompt）消耗的 token 数。"""

    output_tokens: Optional[int] = None
    """输出（completion）消耗的 token 数。"""

    total_tokens: Optional[int] = None
    """input + output，部分 provider 直接返回此字段。"""

    reasoning_tokens: Optional[int] = None
    """
    推理/思考消耗的 token 数，包含在 output_tokens 中。
    OpenAI o1/o3 系列、Anthropic Claude 3.7 extended thinking 支持。
    """

    cached_input_tokens: Optional[int] = None
    """
    命中 prompt cache 的 token 数，包含在 input_tokens 中。
    Anthropic / OpenAI 均有不同形式的 prompt cache 计费策略。
    """

    provider: Dict[str, Any] = {}
    """厂商原始用量字段透传区，例如 Anthropic 的 cache_creation_input_tokens。"""
