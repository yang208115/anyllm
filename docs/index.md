# AnyLLM 开发文档

欢迎阅读 AnyLLM 的开发文档。AnyLLM 是一个通用大模型网关核心库，提供 provider-neutral 的语义层（UIR），支持 OpenAI / Anthropic / Gemini / Bedrock / Ollama 等 LLM API 之间的无损互转。

## 文档索引

### 入门

| 文档 | 内容 |
|------|------|
| [快速上手](quickstart.md) | 安装、配置、第一个请求 |

### 架构与设计

| 文档 | 内容 |
|------|------|
| [架构设计](architecture.md) | 整体架构、核心组件关系、数据流、设计原则 |
| [UIR 模型参考](uir-reference.md) | Universal Intermediate Representation 完整字段参考 |
| [能力矩阵](capabilities.md) | Provider 能力分层（L0/L1/L2）与预设能力声明 |

### 开发指南

| 文档 | 内容 |
|------|------|
| [适配器开发指南](adapters.md) | 如何编写一个新的 provider 适配器 |
| [拦截器开发指南](interceptors.md) | 自定义拦截器的三种方式、内置拦截器详解 |
| [网关使用指南](gateway.md) | AnyLLMGateway 的完整 API 与用法 |
| [转换警告参考](warnings.md) | ConversionWarning 错误码、严重等级、处理策略 |

## 项目结构

```
anyllm/
├── __init__.py                 # 顶层导出
├── gateway.py                  # AnyLLMGateway 网关入口
├── interceptors.py             # 内置拦截器 + FunctionInterceptor
├── schema/                     # UIR 数据模型（Pydantic v2）
│   ├── content.py              # ContentBlock 体系（9 种）
│   ├── message.py              # Message + Role
│   ├── request.py              # UniversalRequest
│   ├── response.py             # UniversalResponse
│   ├── stream.py               # UniversalStreamEvent
│   ├── tools.py                # ToolDef + ToolChoice + ResponseFormat
│   ├── usage.py                # Usage
│   └── warnings.py             # ConversionWarning + ConversionResult
├── adapters/                   # 厂商适配器
│   ├── base.py                 # BaseAdapter + BaseInterceptor + ProviderCapabilities
│   ├── openai_chat.py          # OpenAI Chat Completions
│   ├── anthropic.py            # Anthropic Messages API
│   └── gemini.py               # Google Gemini generateContent API
├── conversion/                 # 转换层
│   ├── converter.py            # UniversalConverter 编排器
│   └── lowering.py             # 降级工具函数
└── capabilities/               # 能力矩阵
    └── matrix.py               # 7 个 provider 的预设能力声明
```

## 技术栈

| 技术 | 版本要求 | 用途 |
|------|---------|------|
| Python | >= 3.10 | 运行环境 |
| Pydantic | >= 2.7.0 | 数据校验、多态反序列化（Discriminated Union） |
| httpx | >= 0.27.0（可选） | 异步 HTTP 客户端（API 调用 + 图片下载） |

## 核心概念速查

| 术语 | 含义 |
|------|------|
| **UIR** | Universal Intermediate Representation — provider-neutral 的语义层 |
| **ContentBlock** | 最小内容单元，9 种类型（text/image/audio/file/thinking/refusal/tool_call/tool_result/provider_block） |
| **Adapter** | 厂商适配器，负责 provider 原始格式 ↔ UIR 的双向转换 |
| **Interceptor** | 拦截器（中间件），在 UIR 层面预处理请求 |
| **Converter** | 转换器编排器，管理适配器和拦截器 |
| **Gateway** | 网关入口，在 Converter 之上增加路由和 HTTP 调用 |
| **ConversionResult** | 转换结果包装器，始终携带 warnings 列表 |
| **ProviderCapabilities** | 能力声明矩阵，描述 provider 支持的功能集合 |
