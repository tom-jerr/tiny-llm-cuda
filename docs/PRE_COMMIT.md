# Pre-commit Hooks 使用指南

本项目使用 [pre-commit](https://pre-commit.com/) 来自动化代码质量检查和格式化。

## 🎯 功能

Pre-commit hooks 会在每次 `git commit` 之前自动运行以下检查：

### Python 代码

- ✅ **Ruff Lint**: 检查代码风格和潜在问题
- ✅ **Ruff Format**: 自动格式化 Python 代码

### C++/CUDA 代码

- ✅ **Clang-Format**: 格式化 C++ 和 CUDA 文件

### 通用检查

- ✅ **YAML/TOML/JSON**: 检查配置文件语法
- ✅ **Merge Conflicts**: 检查合并冲突标记
- ✅ **Large Files**: 防止提交大文件 (>10MB)
- ✅ **Trailing Whitespace**: 移除行尾空白
- ✅ **EOF Fixer**: 确保文件以换行结束
- ✅ **Markdown**: 格式化 Markdown 文件

## 📦 安装

### 方法 1: 使用 PDM (推荐)

```bash
# 安装开发依赖
pdm install -d

# 安装 pre-commit hooks
pdm run pre-commit-install
```

### 方法 2: 使用 pip

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 hooks
pre-commit install
```

## 🚀 使用

### 自动运行 (推荐)

安装完成后，每次 `git commit` 时会自动运行：

```bash
git add .
git commit -m "Your commit message"
# pre-commit hooks 会自动运行并修复问题
```

如果 hooks 修改了文件，你需要：

1. 查看修改
2. 重新 `git add` 修改的文件
3. 再次 `git commit`

### 手动运行

#### 检查所有文件

```bash
# 使用 PDM
pdm run pre-commit-run

# 或直接使用 pre-commit
pre-commit run --all-files
```

#### 检查特定文件

```bash
pre-commit run --files src/layers/attention.py
```

#### 运行特定 hook

```bash
# 只运行 ruff
pre-commit run ruff --all-files

# 只运行格式化
pre-commit run ruff-format --all-files

# 只运行 clang-format
pre-commit run clang-format --all-files
```

### 跳过 hooks (不推荐)

紧急情况下可以跳过 hooks：

```bash
git commit --no-verify -m "Emergency fix"
```

**⚠️ 警告**: 只在紧急情况下使用，并在之后修复问题。

## 🔧 独立工具使用

除了 pre-commit，你也可以单独使用这些工具：

### Ruff

```bash
# Lint 检查
pdm run lint
# 或
ruff check .

# 自动修复
pdm run lint-fix
# 或
ruff check --fix .

# 格式化
pdm run format
# 或
ruff format .
```

### Clang-Format

```bash
# 格式化所有 C++/CUDA 文件
pdm run format-cpp

# 或手动格式化单个文件
clang-format -i src/extensions/ops/vector_add.cu
```

## 📝 配置文件

### .pre-commit-config.yaml

Pre-commit 主配置文件，定义了要运行的 hooks。

### ruff.toml

Ruff linter 和 formatter 的配置：

- 代码风格规则
- 忽略的规则
- 导入排序配置

### .clang-format

C++ 和 CUDA 代码格式化配置：

- 基于 Google 风格
- 缩进、空格、换行规则
- Include 排序

## 📚 扩展阅读

- [Pre-commit 官方文档](https://pre-commit.com/)
- [Ruff 文档](https://docs.astral.sh/ruff/)
- [Clang-Format 文档](https://clang.llvm.org/docs/ClangFormat.html)

## 🤝 贡献

如果你想添加新的 hooks 或修改配置：

1. 编辑 `.pre-commit-config.yaml`
2. 运行 `pre-commit run --all-files` 测试
3. 提交 PR 并说明修改原因

---

**记住**: Pre-commit hooks 是为了帮助我们，而不是阻碍开发。合理配置可以显著提升代码质量！
