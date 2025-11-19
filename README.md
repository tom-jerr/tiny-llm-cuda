# MiniInfer

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**一个从零开始构建的轻量级高性能 LLM 推理引擎**

[特性](#特性) • [快速开始](#快速开始) • [架构](#架构) • [示例](#示例) • [文档](#文档)

</div>

---

## 📝 项目简介

MiniInfer 是一个学习性质的大语言模型推理引擎，从零实现了现代 LLM 推理系统的核心组件。本项目旨在帮助开发者深入理解 LLM 推理的底层机制，包括注意力机制、KV 缓存、批处理调度等关键技术。

### 核心目标

- ⚡ **性能优化**：实现主流推理优化技术
- 🔧 **易于扩展**：模块化设计，便于添加新功能
- 📚 **完整实现**：从基础算子到完整推理流程

## ✨ 特性

### 已实现功能

#### 🧠 核心组件

- ✅ **注意力机制**
  - Multi-Head Attention (MHA)
  - Grouped Query Attention (GQA)
  - 支持因果掩码 (Causal Mask)

- ✅ **位置编码**
  - Rotary Position Embedding (RoPE)
  - 支持传统和非传统模式（Qwen2 使用 non-tranditional）

- ✅ **激活函数**

- ✅ **归一化层**
  - RMSNorm

#### 🚀 推理优化

- ✅ **KV Cache**
  - 单请求 KV Cache
  - 批处理 KV Cache (尾部对齐)
  - 支持动态请求管理

- ✅ **批处理推理**
  - Continuous Batching
  - 动态请求调度
  - 支持多请求并发

- ✅ **CUDA 扩展**
  - C++/CUDA 自定义算子
  - PyTorch C++ 扩展框架
  - 向量加法示例（可扩展更多算子）

#### 🤖 模型支持

- ✅ **Qwen2** 系列模型 (0.5B, 1.5B, 7B)
  > :skull: 7B 未进行测试
  - 完整的模型实现
  - 权重加载与转换
  - 量化支持 (FP16)

### 🔮 计划功能 (详见 [ROADMAP.md](./ROADMAP.md))

- :construction: 项目代码重构中(**重构目标类似 nano-vllm**)
- 🔄 张量并行 (Tensor Parallelism)
- 🔄 流水线并行 (Pipeline Parallelism)
- 🔄 Flash Attention 集成
- 🔄 PagedAttention
- 🔄 推测解码 (Speculative Decoding)
- 🔄 量化支持 (INT8/INT4)
- 🔄 更多模型架构

## 🚀 快速开始

> 💡 **新手推荐**: 查看详细的 [快速入门指南](./docs/QUICKSTART.md) 获取完整教程和常见问题解答。

### 环境要求

- Python 3.10-3.12
- CUDA 11.8+ (用于 GPU 加速)

### 安装

1. **克隆仓库**

```bash
git clone https://github.com/tom-jerr/MiniInfer .git
cd MiniInfer
```

2. **安装依赖**

使用 PDM:

```bash
pip install pdm
pdm install
```

3. **构建 CUDA 扩展** (可选)

```bash
pdm run build-ext
pdm run build-ext-test  # 测试扩展
```

### 基础使用

#### 1. 单个请求推理

```bash
# 使用简单生成 (无 KV Cache)
pdm main
# 或使用带 KV Cache 版本
pdm main-with-kvcache
```

#### 2. 批处理推理

```bash
pdm batch-main
```

#### 3. 运行测试

```bash
# 运行所有测试
pdm test

# 运行特定测试
pdm test --fun gqa # 会测试 unittests/test_gqa.py
```

### 使用示例

:construction: 正在施工

## 🏗️ 架构

### 项目结构

:construction: 正在施工

### 核心设计

:construction: 正在施工

## 🧪 测试

项目包含完整的测试套件：

```bash
# 运行所有测试
pdm run test

# 运行特定测试
# 运行特定测试
pdm test --fun gqa # 会测试 unittests/test_gqa.py
```

## 🗺️ 开发路线

查看 [ROADMAP.md](./ROADMAP.md) 了解详细的开发计划和进度。

### 近期目标 (Q4 2025 - Q1 2026)

- [ ] Flash Attention 集成
- [ ] PagedAttention 实现
- [ ] 张量并行基础支持

### 中期目标 (Q2-Q3 2026)

- [ ] 完整的 Tensor Parallelism
- [ ] Pipeline Parallelism
- [ ] INT8 量化支持

## 📚 文档

- **[快速入门指南](./docs/QUICKSTART.md)** - 5 分钟上手教程，包含常见问题解答
- **[开发路线图](./ROADMAP.md)** - 详细的功能规划和开发进度
- **[贡献指南](./CONTRIBUTING.md)** - 如何为项目做贡献
- **[更新日志](./CHANGELOG.md)** - 版本历史和变更记录
- **[激活函数接口文档](./docs/activation_interface.md)** - 激活函数使用说明

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

本项目在开发过程中参考了以下优秀项目和资源：

### 开源项目

- **[tiny-llm](https://github.com/skyzh/tiny-llm)** - 最初的项目框架代码参考
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** - 重构后项目框架代码参考
- **[vLLM](https://github.com/vllm-project/vllm)** - 高性能 LLM 推理引擎，PagedAttention 和 Continuous Batching 的创新实现给了我们很大启发

### 理论基础

- **Attention Is All You Need** - Transformer 架构的奠基论文
- **GQA: Training Generalized Multi-Query Transformer Models** - Grouped Query Attention 的设计思想
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** - vLLM 的核心技术论文

### 特别致谢

感谢所有为开源 LLM 生态系统做出贡献的开发者和研究者。本项目作为学习性质的实现，旨在帮助更多人理解现代 LLM 推理系统的工作原理。

如果本项目的代码中有任何参考未明确标注来源，请联系我们，我们会立即补充说明。

## 📚 相关资源

- [Transformer 论文](https://arxiv.org/abs/1706.03762)
- [GQA 论文](https://arxiv.org/abs/2305.13245)
- [vLLM 论文](https://arxiv.org/abs/2309.06180)
- [Qwen2 技术报告](https://arxiv.org/abs/2407.10671)

## 📧 联系方式

- 作者: lzy
- Email: tomlzy213@gmail.com
- GitHub: [@tom-jerr](https://github.com/tom-jerr)

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐️ Star！**

Made with ❤️ by [tom-jerr](https://github.com/tom-jerr)

</div>
