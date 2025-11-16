# Changelog

所有项目重要变更都会记录在这个文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中

- Flash Attention 集成
- PagedAttention 实现
- 张量并行支持
- INT8 量化

## [0.1.0] - 2025-11-16

### 新增

- 🎉 项目初始版本发布
- ✨ 完整的 Qwen2 模型实现
- ✨ Multi-Head Attention (MHA) 支持
- ✨ Grouped Query Attention (GQA) 支持
- ✨ Rotary Position Embedding (RoPE) 实现
- ✨ KV Cache 机制
  - 单请求 KV Cache (`TinyKvFullCache`)
  - 批处理 KV Cache (`BatchingKvCache`)
- ✨ Continuous Batching 支持
- ✨ 多种激活函数
  - SiLU/Swish, GELU, ReLU, Leaky ReLU, Tanh, Sigmoid
- ✨ RMSNorm 和 LayerNorm
- ✨ 采样策略
  - Greedy, Temperature, Top-p, Top-k
- ✨ CUDA 扩展框架
  - C++/CUDA 自定义算子示例
- 📚 完整的文档
  - README.md
  - ROADMAP.md
  - CONTRIBUTING.md
- 🧪 测试套件
  - 注意力机制测试
  - 模型推理测试
  - 批处理测试
- 📝 使用示例
  - 激活函数示例
  - 注意力机制示例

### 已知问题

- Flash Attention 尚未集成
- 仅支持单 GPU 推理
- 文档需要进一步完善

## [0.0.1] - 2025-11-01

### 新增

- 🎬 项目初始化
- 📁 基础项目结构
- 🔧 开发工具配置

---

## 版本说明

### [0.1.0] 主要特性

这是 MiniInfer 的首个公开版本，实现了：

1. **完整的推理流程**

   - 从 token 输入到生成输出的完整流程
   - 支持单请求和批处理推理

2. **高效的 KV Cache**

   - 避免重复计算，显著提升推理速度
   - 批处理场景下的智能缓存管理

3. **现代注意力机制**

   - 支持 MHA 和 GQA
   - 灵活的实现切换机制

4. **模块化设计**

   - 清晰的代码结构
   - 易于扩展和定制

5. **教育友好**
   - 详细的代码注释
   - 完整的文档和示例

### 性能指标 (0.1.0)

测试环境: NVIDIA RTX 3090, Qwen2-1.5B

- 单请求推理: ~30 tokens/s (with KV Cache)
- 批处理推理 (batch_size=5): ~40 tokens/s
- 内存占用: ~4GB (FP16)

### 升级指南

#### 从开发版升级到 0.1.0

如果你之前克隆了开发版本：

```bash
git pull origin main
pdm install
pdm run build-ext  # 重新编译扩展
```

### 已知限制

1. **模型支持**: 目前仅支持 Qwen2 系列
2. **硬件**: 单 GPU，尚未支持分布式
3. **优化**: 未集成 Flash Attention
4. **量化**: 仅支持 FP16

这些限制会在后续版本中逐步解决，详见 [ROADMAP.md](./ROADMAP.md)。

---

## 格式说明

- `新增` - 新功能
- `修改` - 现有功能的变更
- `弃用` - 即将移除的功能
- `移除` - 已删除的功能
- `修复` - Bug 修复
- `安全` - 安全相关的修复

[Unreleased]: https://github.com/tom-jerr/MiniInfer /compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tom-jerr/MiniInfer /releases/tag/v0.1.0
[0.0.1]: https://github.com/tom-jerr/MiniInfer /releases/tag/v0.0.1
