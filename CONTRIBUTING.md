# 贡献指南

感谢你对 MiniInfer 项目的关注！我们欢迎任何形式的贡献。

## 🚀 快速开始

1. **Fork 项目**

   ```bash
   # 点击 GitHub 页面右上角的 Fork 按钮
   ```

2. **克隆你的 Fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/MiniInfer .git
   cd MiniInfer
   ```

3. **创建新分支**

   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

4. **安装开发依赖**

   ```bash
   pdm install -d
   ```

5. **设置 Pre-commit Hooks**
   ```bash
   pdm run pre-commit-install
   ```
   这会自动在每次提交前运行代码检查和格式化。详见 [Pre-commit 使用指南](./docs/PRE_COMMIT.md)。

## 📝 开发流程

### 代码规范

- 使用 **Python 3.10+**
- 遵循 **PEP 8** 代码风格
- 使用 **Ruff** 进行代码格式化和 lint
- 添加必要的**类型注解**
- 编写清晰的**文档字符串**

格式化代码：

```bash
# 格式化 Python 代码
pdm run format

# Lint 检查
pdm run lint

# 自动修复 lint 问题
pdm run lint-fix

# 格式化 C++ 代码
pdm run format-cpp

# 运行所有 pre-commit checks
pdm run pre-commit-run
```

### 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型:**

- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 重构（既不是新增功能，也不是修复 bug）
- `perf`: 性能优化
- `test`: 添加测试
- `chore`: 构建过程或辅助工具的变动

**示例:**

```
feat(attention): add Flash Attention support

- Implement Flash Attention v2 integration
- Add CUDA kernels for optimized attention
- Update tests and documentation

Closes #123
```

### 测试要求

- 为新功能添加单元测试
- 确保所有测试通过
- 测试覆盖率尽量达到 80%+

运行测试：

```bash
pdm run test
```

运行特定测试：

```bash
python tests/test_your_feature.py
```

## 🎯 贡献方向

### 优先级高的任务

查看 [ROADMAP.md](./ROADMAP.md) 了解项目规划，优先贡献标记为高优先级的功能。

当前急需贡献的领域：

- 🔥 Flash Attention 集成
- 🔥 PagedAttention 实现
- 🔥 张量并行支持
- 📚 文档完善
- 🧪 测试覆盖率提升

### 寻找任务

1. 查看 [Issues](https://github.com/tom-jerr/MiniInfer /issues) 页面
2. 寻找标签为 `good first issue` 或 `help wanted` 的问题
3. 在 Issue 中评论表示你想处理该问题

## 📋 Pull Request 流程

1. **确保你的代码符合规范**

   ```bash
   pdm run format
   pdm run test
   ```

2. **提交更改**

   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

3. **推送到你的 Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

4. **创建 Pull Request**
   - 在 GitHub 上打开你的 Fork
   - 点击 "New Pull Request"
   - 填写 PR 模板，描述你的更改
   - 链接相关 Issue（如果有）

5. **代码审查**
   - 维护者会审查你的代码
   - 根据反馈进行修改
   - 所有讨论解决后，PR 将被合并

### PR 检查清单

- [ ] 代码遵循项目风格
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] 提交信息清晰明确
- [ ] 没有不相关的更改

## 🐛 报告 Bug

提交 Bug 报告时请包含：

1. **环境信息**
   - Python 版本
   - PyTorch 版本
   - CUDA 版本
   - 操作系统

2. **重现步骤**
   - 清晰的步骤说明
   - 最小可复现代码
   - 期望行为 vs 实际行为

3. **错误信息**
   - 完整的错误堆栈
   - 相关日志

使用 Bug 报告模板创建 Issue。

## 💡 提出新功能

提交功能请求时请包含：

1. **功能描述**
   - 清晰描述想要的功能
   - 说明使用场景

2. **动机**
   - 为什么需要这个功能
   - 当前的替代方案

3. **实现建议**（可选）
   - 你认为应该如何实现
   - 可能的技术方案

## 📚 改进文档

文档贡献同样重要！

- 修复文档中的错误
- 改进现有文档的清晰度
- 添加使用示例
- 翻译文档（欢迎英文版本）

## 🏗️ 项目结构

了解项目结构有助于贡献：

```
src/
├── layers/          # 神经网络层（注意力、激活等）
├── models/          # 模型实现（Qwen2 等）
├── cache/           # KV Cache 和批处理逻辑
├── utils/           # 工具函数
└── extensions/      # CUDA 扩展

tests/               # 测试文件
examples/            # 使用示例
docs/                # 文档
```

## 🤝 行为准则

- 尊重他人
- 提供建设性的反馈
- 接受不同的观点
- 专注于对项目最有利的事情

## 💬 获得帮助

遇到问题？

- 查看 [文档](./README.md)
- 搜索现有 [Issues](https://github.com/tom-jerr/MiniInfer /issues)
- 在 [Discussions](https://github.com/tom-jerr/MiniInfer /discussions) 提问
- 发送邮件至 tomlzy213@gmail.com

## 🎉 认可贡献者

所有贡献者都会在 README 中得到认可。重要贡献将在 Release Notes 中特别感谢。

---

**再次感谢你的贡献！** 🙏

每一个 PR，每一个 Issue，每一条建议都在帮助项目变得更好！
