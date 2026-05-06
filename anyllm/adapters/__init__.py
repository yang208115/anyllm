"""
AnyLLM 厂商适配器包。

基类：
  BaseAdapter          — 所有适配器必须继承的抽象基类
  BaseInterceptor      — 所有拦截器（中间件）必须继承的抽象基类
  ProviderCapabilities — 声明 provider 支持的能力集合

内置适配器：
  OpenAIChatAdapter    — OpenAI Chat Completions API
  AnthropicAdapter     — Anthropic Messages API
"""

from anyllm.adapters.anthropic import AnthropicAdapter
from anyllm.adapters.base import BaseAdapter, BaseInterceptor, ProviderCapabilities
from anyllm.adapters.gemini import GeminiAdapter
from anyllm.adapters.openai_chat import OpenAIChatAdapter

__all__ = [
    "BaseAdapter",
    "BaseInterceptor",
    "ProviderCapabilities",
    "OpenAIChatAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
]
