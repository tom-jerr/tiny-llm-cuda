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

MiniInfer 是一个教育性质的大语言模型推理引擎，从零实现了现代 LLM 推理系统的核心组件。本项目旨在帮助开发者深入理解 LLM 推理的底层机制，包括注意力机制、KV 缓存、批处理调度等关键技术。

### 核心目标

- 🎯 **教育导向**：清晰的代码结构和详细的注释
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
  - 多种实现方式可切换

- ✅ **位置编码**
  - Rotary Position Embedding (RoPE)
  - 支持传统和非传统模式（Qwen2 使用 non-tranditional）

- ✅ **激活函数**
  - SiLU / Swish
  - GELU / ReLU / Leaky ReLU
  - Tanh / Sigmoid
  - 统一接口，易于扩展

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
- 8GB+ GPU 显存（推荐）

### 安装

1. **克隆仓库**

```bash
git clone https://github.com/tom-jerr/MiniInfer .git
cd MiniInfer
```

2. **安装依赖**

使用 PDM (推荐):

```bash
pip install pdm
pdm install
```

或使用 pip:

```bash
pip install torch>=2.6.0 transformers>=4.51.0 flash-attn>=2.8.3
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
python main.py --model Qwen/Qwen2-1.5B --loader v1 --prompt "介绍一下大语言模型"

# 使用 KV Cache 加速
python main.py --model Qwen/Qwen2-1.5B --loader v2 --prompt "介绍一下大语言模型"
```

#### 2. 批处理推理

```bash
python batch-main.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --batch-size 5 \
  --prefill-step 128 \
  --max-seq-len 512
```

#### 3. 运行测试

```bash
# 运行所有测试
pdm run test

# 运行特定测试
python tests/test_attention.py
python tests/test_qwen2.py
python tests/test_batching.py
```

### 使用示例

查看 `examples/` 目录获取更多示例：

```bash
# 激活函数使用示例
python examples/activation_usage.py

# 注意力机制使用示例
python examples/attention_usage.py
```

## 🏗️ 架构

### 项目结构

```
MiniInfer /
├── src/
│   ├── layers/              # 神经网络层实现
│   │   ├── activation.py    # 激活函数
│   │   ├── attention.py     # 注意力机制
│   │   ├── embedding.py     # 嵌入层
│   │   ├── layernorm.py     # 归一化层
│   │   ├── linear.py        # 线性层
│   │   ├── position_encoding.py  # 位置编码
│   │   └── sampler.py       # 采样器
│   ├── models/              # 模型实现
│   │   ├── qwen2.py         # Qwen2 模型
│   │   └── configs/         # 模型配置
│   ├── cache/               # KV Cache 实现
│   │   ├── kv_cache.py      # KV Cache 核心
│   │   ├── generate.py      # 生成逻辑
│   │   └── request.py       # 请求管理与批处理
│   ├── utils/               # 工具函数
│   │   ├── model_utils.py   # 模型工具
│   │   └── quantize.py      # 量化工具
│   └── extensions/          # CUDA 扩展
│       ├── bindings.cpp     # Python 绑定
│       ├── setup.py         # 编译配置
│       └── ops/             # CUDA 算子
│           ├── vector_add.cu
│           └── vector_add.h
├── tests/                   # 测试文件
├── examples/                # 使用示例
├── main.py                  # 单请求推理入口
├── batch-main.py            # 批处理推理入口
└── pyproject.toml           # 项目配置
```

### 核心设计

#### 1. 模块化注意力实现

支持多种注意力实现，通过字符串选择：

```python
from src.layers.attention import scaled_dot_product_attention

# 使用 GQA
output = scaled_dot_product_attention(
    query, key, value,
    implementation="gqa",
    mask="causal"
)

# 使用简单实现
output = scaled_dot_product_attention(
    query, key, value,
    implementation="simple"
)
```

#### 2. 灵活的 KV Cache

- **单请求 Cache**: `TinyKvFullCache` - 用于单个请求的 KV 缓存
- **批处理 Cache**: `BatchingKvCache` - 支持多请求并发，尾部对齐设计

```python
# 批处理 KV Cache 尾部对齐示例
# batched_keys[i, :, (S-S_i):S, :] = keys[i, :, :, :]
# 使得不同长度序列可以共享统一的因果遮罩
```

#### 3. Continuous Batching

实现了类似 vLLM 的连续批处理调度：

- Prefill 和 Decode 阶段分离
- 动态请求添加与移除
- 高效的批处理调度
- 支持可配置的 prefill 步长

## 📊 示例

### 注意力机制选择

```python
from src.layers.attention import get_attention, scaled_dot_product_attention

# 方法1: 直接获取实现
attn_fn = get_attention("gqa")
output = attn_fn(query, key, value, scale=0.125, mask="causal")

# 方法2: 使用统一接口
output = scaled_dot_product_attention(
    query, key, value,
    scale=0.125,
    mask="causal",
    implementation="gqa"
)
```

### 激活函数使用

```python
from src.layers.activation import get_activation, apply_activation

# 方法1: 获取激活函数
silu = get_activation("silu")
output = silu(x)

# 方法2: 直接应用
output = apply_activation(x, activation="silu")

# 带参数的激活函数
output = apply_activation(x, activation="leaky_relu", negative_slope=0.1)
```

### 批处理推理

```python
from src.cache.request import batch_generate
from src.models.qwen2 import Qwen2ModelV2

# 准备多个 prompts
prompts = [
    "What is the capital of France?",
    "介绍一下上海",
    "Explain quantum computing",
]

# 批处理生成
results = batch_generate(
    model=tiny_llm_model,
    tokenizer=tokenizer,
    prompts=prompts,
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
)

# 输出结果
for prompt_idx, text in results:
    print(f"Prompt {prompt_idx}: {text}")
```

## 🧪 测试

项目包含完整的测试套件：

```bash
# 运行所有测试
pdm run test

# 运行特定测试
python tests/test_attention.py      # 注意力机制测试
python tests/test_gqa.py            # GQA 测试
python tests/test_mha.py            # MHA 测试
python tests/test_rope.py           # RoPE 测试
python tests/test_qwen2.py          # Qwen2 模型测试
python tests/test_batching.py       # 批处理测试
```

## 🛠️ 开发

### PDM 脚本

```bash
pdm run main         # 运行主程序
pdm run main-v1      # 使用 v1 loader
pdm run main-v2      # 使用 v2 loader (KV Cache)
pdm run batch-main   # 批处理推理
pdm run test         # 运行测试
pdm run format       # 代码格式化
pdm run build-ext    # 构建 CUDA 扩展
```

### 添加新的激活函数

```python
# 在 src/layers/activation.py 中
def my_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# 注册到全局字典
ACTIVATION_IMPLEMENTATIONS["my_act"] = my_activation
```

### 添加新的注意力实现

```python
# 在 src/layers/attention.py 中
def my_attention(query, key, value, scale=None, mask=None):
    # 自定义实现
    pass

# 注册
ATTENTION_IMPLEMENTATIONS["my_attn"] = my_attention
```

## 📈 性能

### 批处理效率

使用 Continuous Batching 可以显著提升吞吐量：

- 单请求推理: ~10 tokens/s
- 批处理 (batch_size=5): ~40 tokens/s (4x 提升)
- 内存利用率提升: 2-3x

### KV Cache 加速

- 无 KV Cache: ~10 tokens/s
- 有 KV Cache: ~30-50 tokens/s (3-5x 提升)

_注: 性能数据基于 NVIDIA RTX 3090, Qwen2-1.5B 模型_

## 🗺️ 开发路线

查看 [ROADMAP.md](./ROADMAP.md) 了解详细的开发计划和进度。

### 近期目标 (Q1 2026)

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

- **[vLLM](https://github.com/vllm-project/vllm)** - 高性能 LLM 推理引擎，PagedAttention 和 Continuous Batching 的创新实现给了我们很大启发
- **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** - 高效注意力实现的标杆，为我们的优化方向提供了重要参考
- **[Transformers](https://github.com/huggingface/transformers)** - Hugging Face 的 Transformers 库，提供了完善的模型实现和权重加载方案
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** - NVIDIA 的推理优化方案，CUDA kernel 优化思路值得学习

### 理论基础

- **Attention Is All You Need** - Transformer 架构的奠基论文
- **GQA: Training Generalized Multi-Query Transformer Models** - Grouped Query Attention 的设计思想
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** - vLLM 的核心技术论文
- **FlashAttention: Fast and Memory-Efficient Exact Attention** - Flash Attention 算法

### 特别致谢

感谢所有为开源 LLM 生态系统做出贡献的开发者和研究者。本项目作为教育性质的实现，旨在帮助更多人理解现代 LLM 推理系统的工作原理。

如果本项目的代码中有任何参考未明确标注来源，请联系我们，我们会立即补充说明。

## 📚 相关资源

- [Transformer 论文](https://arxiv.org/abs/1706.03762)
- [GQA 论文](https://arxiv.org/abs/2305.13245)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
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
