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
- ✨ Continuous Batching 支持(Padding 实现)
- ✨ 多种激活函数
  - SiLU/Swish, GELU, ReLU, Leaky ReLU, Tanh, Sigmoid
- ✨ RMSNorm
- ✨ 采样策略
  - Greedy, Temperature, Top-p, Top-k
- ✨ CUDA 扩展框架
  - C++/CUDA 自定义算子示例
- 📚 完整的文档
  - README.md
  - ROADMAP.md
  - CONTRIBUTING.md
- 🧪 测试套件
  - 单元测试
  - 性能测试；gms8k 基准测试

### 已知问题

- Flash Attention 尚未集成，仅在 test 中进行测试
- 仅支持单 GPU 推理
- 文档需要进一步完善

---

## 版本说明

### [0.1.0] 主要特性

这是 MiniInfer 的首个公开版本，实现了：

1. **完整的推理流程**
   - 支持单请求和批处理推理
2. **实现 KV Cache**
3. **支持 GQA**

### 性能指标 (0.1.0)

测试环境: NVIDIA RTX 3080ti, Qwen2-1.5B

- :construction: 暂无

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
