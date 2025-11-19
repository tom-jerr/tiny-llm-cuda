# 🗺️ MiniInfer 开发路线图

本文档详细描述了 MiniInfer 项目的开发规划，包括已完成的功能、正在进行的工作以及未来的开发计划。

## 📊 总体进度

```
基础设施    ████████████████████ 100%
核心功能    ██████████████░░░░░░  70%
推理优化    ████████░░░░░░░░░░░░  40%
分布式训练  ██░░░░░░░░░░░░░░░░░░  10%
生产就绪    ████░░░░░░░░░░░░░░░░  20%
```

---

## ✅ 已完成功能 (Phase 0 & 1)

- [x] **项目结构搭建**
  - [x] 模块化目录设计
  - [x] PDM 包管理配置
  - [x] 开发工具链设置 (pytest, ruff)
- [x] **基础算子实现**
  - [x] 线性层 (Linear)
  - [x] 激活函数
  - [x] 归一化层
    - [x] RMSNorm
- [x] **CUDA 扩展框架**
  - [x] PyTorch C++ 扩展设置
  - [x] CUDA kernel 示例 (vector_add)
  - [x] Python bindings

- [x] **注意力机制**
  - [x] Multi-Head Attention (MHA)
  - [x] Grouped Query Attention (GQA)
  - [x] 因果掩码 (Causal Mask)
- [x] **位置编码**
  - [x] Rotary Position Embedding (RoPE)
  - [x] 传统 RoPE 实现
  - [x] Qwen2 风格 RoPE
  - [x] 支持 offset 和 slice
- [x] **KV Cache**
  - [x] 单请求完整 KV Cache (`TinyKvFullCache`)
  - [x] 批处理 KV Cache (`BatchingKvCache`)
- [x] **模型实现**
  - [x] Qwen2
- [x] **采样策略**
  - [x] Greedy Sampling
  - [x] Temperature Sampling
  - [x] Top-p (Nucleus) Sampling
  - [x] Top-k Sampling

- [x] **Continuous Batching 基础**
  - [x] 请求管理系统 (`Request` 类)
  - [x] Prefill/Decode 阶段分离
  - [x] 动态请求添加/移除
  - [x] 批处理生成接口

---

## 待实现功能

### PagedAttention 实现 (Q4 2025 - Q1 2026)

**优先级**: ⭐⭐⭐⭐⭐

灵感来自 vLLM 和 SGLang，通过分页管理 KV Cache 以及 Radix Cache 显著提升内存利用率。

- [ ] **核心实现**
  - [ ] 页表管理系统
  - [ ] 物理/逻辑页映射
  - [ ] KV Cache 分页存储
  - [ ] Copy-on-Write 机制

- [ ] **PagedAttention Kernel**
  - [ ] CUDA kernel 实现
  - [ ] 非连续内存访问优化
  - [ ] 与 Flash Attention 结合

- [ ] **前缀识别与匹配**
  - [ ] 前缀树 (Trie) 结构
  - [ ] 最长公共前缀匹配

- [ ] **Cache 管理**
  - [ ] 前缀 KV Cache 存储
  - [ ] LRU 淘汰策略
  - [ ] 前缀共享

### 推测解码 (Speculative Decoding) (Q1 - Q2 2026)

**优先级**: ⭐⭐⭐⭐⭐

使用小模型加速大模型推理。

- [ ] **基础实现**
  - [ ] Draft 模型推理
  - [ ] Target 模型验证
  - [ ] Token 接受/拒绝逻辑
  - [ ] 多 token 并行验证

- [ ] **优化策略**
  - [ ] 自适应草稿长度
  - [ ] Draft 模型选择策略
  - [ ] 树形推测 (Tree Attention)

### Flash Attention 集成 (Q4 2025 - Q1 2026)

**优先级**: ⭐⭐⭐⭐⭐

- [ ] **Flash Attention v2 集成**
  - [ ] 安装与配置 flash-attn 库
  - [ ] 适配 Flash Attention API
  - [ ] GQA 支持
  - [ ] 性能测试与对比

- [ ] **自定义 Flash Attention Kernel**
  - [ ] CUDA kernel 实现
  - [ ] Tiling 优化
  - [ ] 寄存器优化
  - [ ] 与 PyTorch 实现性能对比

### 量化支持 (Q2 2026)

**优先级**: ⭐⭐⭐⭐

- [ ] **INT8 量化**
  - [ ] 权重量化
  - [ ] 激活量化
  - [ ] 对称/非对称量化
  - [ ] 量化感知训练

- [ ] **INT4/GPTQ 量化**
  - [ ] GPTQ 算法实现
  - [ ] Group-wise 量化
  - [ ] 自定义 CUDA kernel

- [ ] **FP8 量化** (如果硬件支持)
  - [ ] FP8 E4M3/E5M2 格式
  - [ ] 混合精度推理

### 张量并行 (Tensor Parallelism) (Q1 - Q2 2026)

**优先级**: ⭐⭐⭐⭐⭐

- [ ] **基础实现**
  - [ ] 列并行 (Column Parallel) Linear
  - [ ] 行并行 (Row Parallel) Linear
  - [ ] Attention 层并行切分
  - [ ] MLP 层并行切分
  - [ ] All-Reduce / All-Gather 通信

- [ ] **优化策略**
  - [ ] 通信与计算重叠
  - [ ] 1D / 2D / 3D 并行策略
  - [ ] 自动并行切分

- [ ] **集成与测试**
  - [ ] 单节点多 GPU 测试
  - [ ] Qwen2 模型 TP 支持
  - [ ] 性能测试

**技术栈**: `torch.distributed`, NCCL

### 流水线并行 (Pipeline Parallelism) (Q2 - Q3 2026)

**优先级**: ⭐⭐⭐⭐

- [ ] **基础实现**
  - [ ] 模型层切分
  - [ ] 微批处理 (Micro-batching)
  - [ ] 流水线调度
  - [ ] GPipe / PipeDream 策略

- [ ] **优化**
  - [ ] 1F1B 调度
  - [ ] 气泡时间优化
  - [ ] 内存优化

### 专家混合并行 (Expert Parallelism) (Q3 - Q4 2026)

**优先级**: ⭐⭐⭐

为 MoE 模型提供支持。

- [ ] **基础支持**
  - [ ] 专家路由
  - [ ] 专家并行切分
  - [ ] 负载均衡

---

## 后续 Serving 支持 (Q2 - Q4 2026)

### 推理服务 (Serving)

- [ ] **HTTP API**
  - [ ] RESTful API 设计
  - [ ] OpenAI 兼容接口
  - [ ] Streaming 响应
  - [ ] 异步请求处理

- [ ] **gRPC 支持**
  - [ ] Protocol Buffers 定义
  - [ ] 高性能 RPC 服务

- [ ] **批处理优化**
  - [ ] 动态批处理
  - [ ] 请求排队
  - [ ] 超时处理

### 5.2 监控与日志 (Q3 2026)

- [ ] **性能监控**
  - [ ] 请求延迟统计
  - [ ] 吞吐量监控
  - [ ] GPU 利用率监控
  - [ ] 内存使用监控

- [ ] **日志系统**
  - [ ] 结构化日志
  - [ ] 日志级别控制
  - [ ] 分布式追踪

- [ ] **可视化面板**
  - [ ] Prometheus 集成
  - [ ] Grafana Dashboard

### 5.3 部署支持 (Q3 - Q4 2026)

- [ ] **容器化**
  - [ ] Docker 镜像
  - [ ] CUDA 支持
  - [ ] 多阶段构建优化

- [ ] **编排支持**
  - [ ] Kubernetes 部署配置
  - [ ] Helm Charts
  - [ ] 自动扩缩容

- [ ] **模型格式支持**
  - [ ] GGUF 格式
  - [ ] SafeTensors
  - [ ] 自定义格式

---

## 📅 时间线

```
2025 Q4: Phase 2 - 核心优化基础
├── 🔄 Continuous Batching 优化持续进行
├── 🚀 Flash Attention 集成启动
└── 🚀 PagedAttention 基础实现启动

2026 Q1: Phase 2-3 - 高性能推理优化
├── ✅ Flash Attention 集成完成
├── ✅ PagedAttention 基础实现完成
├── 🚀 Tensor Parallelism 启动
└── 🚀 推测解码启动

2026 Q2: Phase 3-4 - 分布式与高级优化
├── ✅ PagedAttention 完整实现
├── ✅ Tensor Parallelism 基础支持
├── 🔄 Speculative Decoding 持续优化
├── 🚀 Pipeline Parallelism 启动
├── 🚀 INT8 量化启动
└── 🚀 推理服务 API 启动

2026 Q3: Phase 4-5 - 生产特性
├── ✅ Pipeline Parallelism 完成
├── ✅ 推理服务 API 完成
├── ✅ 监控系统完成
├── 🔄 INT4 量化
├── 🚀 Expert Parallelism 启动
└── 🚀 部署工具链启动

2026 Q4: Phase 5-6 - 生产就绪与扩展
├── ✅ 部署工具链完成
├── ✅ Expert Parallelism 完成
├── 🔄 更多模型支持
└── 🔄 长上下文优化
```

---

## 📝 参考文献与资源

### 论文

- **Attention Is All You Need** - Transformer 原论文
- **Flash Attention** - 高效注意力实现
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** - vLLM 核心技术
- **Fast Inference from Transformers via Speculative Decoding** - 推测解码
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** - GQA

### 开源项目

- **tiny-llm** - 最初的项目框架代码参考
- **nano-vllm** - 重构后项目框架代码参考

---

## 🤝 贡献指南

欢迎对 ROADMAP 中的任何项目做出贡献！

### 如何贡献

1. 在 Issue 中声明你要实现该功能
2. Fork 项目并创建分支
3. 实现功能并添加测试
4. 提交 PR，更新 ROADMAP 状态

### 优先级说明

- ⭐⭐⭐⭐⭐ 最高优先级（核心功能）
- ⭐⭐⭐⭐ 高优先级（重要优化）
- ⭐⭐⭐ 中等优先级（增强功能）
- ⭐⭐ 低优先级（扩展功能）
- ⭐ 可选功能

---

## 📮 反馈

对 ROADMAP 有任何建议？请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至 tomlzy213@gmail.com
- 参与 Discussions 讨论

---

<div align="center">

**让我们一起构建一个强大的 LLM 推理引擎！** 🚀

更新时间: 2025-11-19

</div>
