#!/bin/bash
# Pre-commit hooks 快速设置脚本

set -e  # 遇到错误立即退出

echo "🚀 MiniInfer  Pre-commit Hooks 设置脚本"
echo "=========================================="
echo ""

# 检查是否在项目根目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查 Python 版本
echo "📋 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python 版本: $python_version"

# 检查是否安装了 pdm
if command -v pdm &> /dev/null; then
    echo "✅ PDM 已安装"
    use_pdm=true
else
    echo "⚠️  PDM 未安装，将使用 pip"
    use_pdm=false
fi

# 安装依赖
echo ""
echo "📦 安装依赖..."
if [ "$use_pdm" = true ]; then
    pdm install -d
else
    pip install pre-commit ruff
fi

# 检查是否安装了 clang-format
echo ""
echo "🔍 检查 clang-format..."
if command -v clang-format &> /dev/null; then
    clang_version=$(clang-format --version)
    echo "✅ clang-format 已安装: $clang_version"
else
    echo "⚠️  clang-format 未安装"
    echo "   Ubuntu/Debian: sudo apt-get install clang-format"
    echo "   macOS: brew install clang-format"
    echo "   跳过 C++ 格式化: SKIP=clang-format git commit"
fi

# 安装 pre-commit hooks
echo ""
echo "🔧 安装 pre-commit hooks..."
if [ "$use_pdm" = true ]; then
    pdm run pre-commit install
else
    pre-commit install
fi

# 运行一次检查
echo ""
echo "🧪 运行首次检查..."
echo "   (这可能需要几分钟，因为需要下载 hooks)"
if [ "$use_pdm" = true ]; then
    pdm run pre-commit run --all-files || true
else
    pre-commit run --all-files || true
fi

echo ""
echo "✅ 设置完成！"
echo ""
echo "📚 使用指南:"
echo "   - 查看详细文档: cat docs/PRE_COMMIT.md"
echo "   - 手动运行检查: pdm run pre-commit-run"
echo "   - 格式化代码: pdm run format"
echo "   - Lint 检查: pdm run lint"
echo ""
echo "🎉 现在每次 git commit 都会自动运行代码检查！"
