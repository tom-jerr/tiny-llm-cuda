# 快速入门指南

本指南将帮助你快速上手 MiniInfer 项目。

## 📋 前置要求

### 硬件要求

- **CPU**: 任意现代 CPU
- **GPU**: NVIDIA GPU with CUDA support (推荐 8GB+ 显存)
  - 最小: GTX 1060 6GB (仅支持小模型)
  - 推荐: RTX 3090, A100, H100
- **内存**: 16GB+ RAM

### 软件要求

- **操作系统**: Linux (推荐), Windows (WSL2), macOS (仅 CPU)
- **Python**: 3.10, 3.11, 或 3.12
- **CUDA**: 11.8+ (用于 GPU 加速)
- **Git**: 版本控制

## 🚀 快速开始 (5 分钟)

### 步骤 1: 克隆项目

```bash
git clone https://github.com/tom-jerr/MiniInfer .git
cd MiniInfer
```

### 步骤 2: 安装依赖

**选项 A: 使用 PDM (推荐)**

```bash
# 安装 PDM
pip install pdm

# 安装项目依赖
pdm install
```

**选项 B: 使用 pip**

```bash
pip install torch>=2.6.0 transformers>=4.51.0 accelerate>=1.11.0
pip install flash-attn>=2.8.3  # 可选，用于 Flash Attention
```

### 步骤 3: 运行第一个示例

```bash
# 使用 PDM
pdm run main --model Qwen/Qwen2-1.5B --prompt "Hello, how are you?"

# 或直接使用 Python
python main.py --model Qwen/Qwen2-1.5B --prompt "Hello, how are you?"
```

第一次运行会自动下载模型（约 3GB），请耐心等待。

## 💡 基础使用

### 1. 单个请求推理

#### 简单生成（无 KV Cache）

```bash
python main.py \
  --model Qwen/Qwen2-1.5B \
  --loader v1 \
  --prompt "介绍一下人工智能"
```

#### 使用 KV Cache 加速

```bash
python main.py \
  --model Qwen/Qwen2-1.5B \
  --loader v2 \
  --prompt "介绍一下人工智能"
```

**性能对比**: KV Cache 可以提升 3-5x 生成速度！

### 2. 批处理推理

```bash
python batch-main.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --batch-size 5 \
  --prefill-step 128 \
  --max-seq-len 512 \
  --max-new-tokens 100
```

### 3. 自定义采样参数

```bash
python main.py \
  --model Qwen/Qwen2-1.5B \
  --prompt "Tell me a story" \
  --sampler-temp 0.8 \
  --sampler-top-p 0.95 \
  --sampler-top-k 50
```

**参数说明**:

- `--sampler-temp`: 温度参数，越高输出越随机 (0.0-2.0)
- `--sampler-top-p`: Nucleus sampling 阈值 (0.0-1.0)
- `--sampler-top-k`: Top-K sampling 数量

### 4. 使用不同模型

```bash
# Qwen2 0.5B (最快，适合测试)
python main.py --model Qwen/Qwen2-0.5B-Instruct

# Qwen2 1.5B (平衡)
python main.py --model Qwen/Qwen2-1.5B

# Qwen2 7B (需要 16GB+ 显存)
python main.py --model Qwen/Qwen2-7B
```

## 🧪 运行测试

### 运行所有测试

```bash
pdm run test
# 或
python scripts/dev-tools.py test
```

### 运行特定测试

```bash
# 测试注意力机制
python tests/test_attention.py

# 测试 Qwen2 模型
python tests/test_qwen2.py

# 测试批处理
python tests/test_batching.py

# 测试 GQA
python tests/test_gqa.py

# 测试 RoPE
python tests/test_rope.py
```

## 📚 学习示例

项目提供了详细的使用示例，帮助理解各个组件：

### 激活函数示例

```bash
python examples/activation_usage.py
```

你将学到：

- 如何使用不同的激活函数
- 激活函数的统一接口
- 在模型中集成激活函数

### 注意力机制示例

```bash
python examples/attention_usage.py
```

你将学到：

- MHA 和 GQA 的使用
- 如何切换不同的注意力实现
- 因果遮罩的应用

## 🔧 开发工具

### 代码格式化

```bash
pdm run format
```

使用 Ruff 格式化所有 Python 代码。

### 构建 CUDA 扩展

```bash
# 构建扩展
pdm run build-ext

# 测试扩展
pdm run build-ext-test

# 清理构建文件
pdm run clean-ext
```

## 🐛 常见问题

### Q1: 下载模型太慢怎么办？

**A**: 使用国内镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python main.py --model Qwen/Qwen2-1.5B
```

### Q2: CUDA Out of Memory 错误

**A**: 尝试以下方法：

1. 使用更小的模型（0.5B 而不是 7B）
2. 减少批处理大小
3. 减少 `max-seq-len`

```bash
python batch-main.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --batch-size 2 \
  --max-seq-len 256
```

### Q3: Flash Attention 安装失败

**A**: Flash Attention 是可选依赖，安装失败不影响基础功能：

```bash
# 跳过 flash-attn 安装
pip install torch transformers accelerate
```

或使用预编译轮子：

```bash
pip install flash-attn --no-build-isolation
```

### Q4: Windows 上如何运行？

**A**: 推荐使用 WSL2：

1. 安装 WSL2 和 Ubuntu
2. 在 WSL2 中安装 CUDA
3. 按照 Linux 步骤安装项目

### Q5: 如何使用 CPU 运行？

**A**: 添加 `--device cpu` 参数：

```bash
python main.py --model Qwen/Qwen2-0.5B --device cpu
```

注意：CPU 推理速度会很慢。

## 📊 性能优化建议

### 1. 使用 KV Cache

```bash
# 慢 (10 tokens/s)
python main.py --loader v1

# 快 (30-50 tokens/s)
python main.py --loader v2
```

### 2. 批处理推理

批处理可以提升吞吐量：

```bash
python batch-main.py --batch-size 5  # 4x 吞吐量提升
```

### 3. 使用 FP16

默认使用 FP16，速度快且内存占用小。

### 4. 调整 Prefill 步长

```bash
# 小步长: 更灵活，但可能更慢
python batch-main.py --prefill-step 64

# 大步长: 更快，但灵活性降低
python batch-main.py --prefill-step 256
```

## 🎯 下一步

现在你已经掌握了基础使用，可以：

1. **深入学习**: 阅读 [README.md](./README.md) 了解架构设计
2. **贡献代码**: 查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 和 [ROADMAP.md](./ROADMAP.md)
3. **探索代码**: 阅读 `src/` 目录下的源码和注释
4. **实验优化**: 尝试实现 ROADMAP 中的功能

## 📖 推荐阅读

- [项目架构说明](./README.md#架构)
- [开发路线图](./ROADMAP.md)
- [贡献指南](./CONTRIBUTING.md)
- [API 文档](./docs/)

## 💬 获得帮助

遇到问题？

- 查看 [Issues](https://github.com/tom-jerr/MiniInfer /issues)
- 提问在 [Discussions](https://github.com/tom-jerr/MiniInfer /discussions)
- 发送邮件至 tomlzy213@gmail.com

---

**祝你使用愉快！** 🎉

如果觉得项目有帮助，别忘了给个 ⭐️ Star！
