"""
演示如何使用新的attention实现选择API
"""

import sys

import torch

sys.path.append('..')

from src.layers.attention import (
    ATTENTION_IMPLEMENTATIONS,
    get_attention_implementation,
    scaled_dot_product_attention,
)


def demo_attention_implementations():
    """演示不同的attention实现"""

    # 创建示例数据
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print("=" * 60)
    print("可用的Attention实现:")
    print("=" * 60)
    for name in ATTENTION_IMPLEMENTATIONS.keys():
        print(f"  - {name}")
    print()

    # 方法1: 使用统一接口
    print("=" * 60)
    print("方法1: 使用统一的 scaled_dot_product_attention 接口")
    print("=" * 60)

    # 使用GQA实现
    output_gqa = scaled_dot_product_attention(
        query, key, value,
        scale=1.0 / (head_dim ** 0.5),
        mask="causal",
        implementation="gqa"
    )
    print(f"使用 'gqa' 实现: output shape = {output_gqa.shape}")

    # 使用简单实现
    output_simple = scaled_dot_product_attention(
        query, key, value,
        implementation="simple"
    )
    print(f"使用 'simple' 实现: output shape = {output_simple.shape}")

    # 使用参考实现
    output_ref = scaled_dot_product_attention(
        query, key, value,
        implementation="reference"
    )
    print(f"使用 'reference' 实现: output shape = {output_ref.shape}")
    print()

    # 方法2: 直接获取实现函数
    print("=" * 60)
    print("方法2: 使用 get_attention_implementation 获取函数")
    print("=" * 60)

    attn_fn = get_attention_implementation("grouped")
    output = attn_fn(query, key, value, scale=0.125, mask="causal")
    print(f"获取 'grouped' 实现并调用: output shape = {output.shape}")
    print()

    # 演示错误处理
    print("=" * 60)
    print("错误处理演示:")
    print("=" * 60)
    try:
        scaled_dot_product_attention(
            query, key, value,
            implementation="unknown_implementation"
        )
    except ValueError as e:
        print(f"✓ 捕获到预期的错误: {e}")
    print()


def demo_gqa_attention():
    """演示GQA (Grouped Query Attention)"""

    print("=" * 60)
    print("GQA (Grouped Query Attention) 演示")
    print("=" * 60)

    batch_size = 2
    num_query_heads = 32  # 查询头数
    num_kv_heads = 8      # KV头数（比查询头数少）
    seq_len = 10
    head_dim = 64

    query = torch.randn(batch_size, num_query_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    print(f"Query shape: {query.shape}  (num_heads={num_query_heads})")
    print(f"Key shape:   {key.shape}  (num_heads={num_kv_heads})")
    print(f"Value shape: {value.shape}  (num_heads={num_kv_heads})")
    print(f"Ratio: {num_query_heads // num_kv_heads} query heads per KV head")
    print()

    output = scaled_dot_product_attention(
        query, key, value,
        implementation="gqa",
        mask="causal"
    )

    print(f"Output shape: {output.shape}")
    print("✓ GQA成功处理不同数量的query和KV heads")
    print()


def demo_dynamic_selection():
    """演示动态选择attention实现"""

    print("=" * 60)
    print("动态选择Attention实现")
    print("=" * 60)

    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 模拟根据配置选择实现
    config = {
        "attention_type": "grouped"  # 可以从配置文件读取
    }

    print(f"从配置读取: attention_type = '{config['attention_type']}'")

    output = scaled_dot_product_attention(
        query, key, value,
        implementation=config["attention_type"]
    )

    print(f"使用配置的实现计算attention: output shape = {output.shape}")
    print()


if __name__ == "__main__":
    demo_attention_implementations()
    demo_gqa_attention()
    demo_dynamic_selection()

    print("=" * 60)
    print("所有演示完成！")
    print("=" * 60)
