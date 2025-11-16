"""
示例：使用统一的activation函数接口

这个示例展示了如何使用 get_activation 和 apply_activation 函数
来灵活地选择和应用不同的激活函数。
"""

import sys

import torch

sys.path.append("..")

from src.layers.activation import apply_activation, get_activation


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("基本使用示例")
    print("=" * 60)

    x = torch.randn(2, 3)
    print(f"输入张量:\n{x}\n")

    # 方法1: 使用 get_activation 获取函数
    silu_fn = get_activation("silu")
    output1 = silu_fn(x)
    print(f"SiLU 激活输出:\n{output1}\n")

    # 方法2: 使用 apply_activation 统一接口
    output2 = apply_activation(x, activation="relu")
    print(f"ReLU 激活输出:\n{output2}\n")

    # GELU 激活
    output3 = apply_activation(x, activation="gelu")
    print(f"GELU 激活输出:\n{output3}\n")


def example_with_parameters():
    """带参数的激活函数示例"""
    print("=" * 60)
    print("带参数的激活函数示例")
    print("=" * 60)

    x = torch.randn(2, 3)
    print(f"输入张量:\n{x}\n")

    # Leaky ReLU with custom negative_slope
    output1 = apply_activation(x, activation="leaky_relu", negative_slope=0.1)
    print(f"Leaky ReLU (slope=0.1) 输出:\n{output1}\n")

    output2 = apply_activation(x, activation="leaky_relu", negative_slope=0.2)
    print(f"Leaky ReLU (slope=0.2) 输出:\n{output2}\n")


def example_in_model():
    """在模型中使用示例"""
    print("=" * 60)
    print("在模型中使用示例")
    print("=" * 60)

    class SimpleFFN(torch.nn.Module):
        def __init__(self, dim, hidden_dim, activation="silu"):
            super().__init__()
            self.w1 = torch.nn.Linear(dim, hidden_dim)
            self.w2 = torch.nn.Linear(hidden_dim, dim)
            self.activation = activation

        def forward(self, x):
            # 使用统一的激活函数接口
            return self.w2(apply_activation(self.w1(x), activation=self.activation))

    # 创建不同激活函数的模型
    x = torch.randn(2, 8)

    model_silu = SimpleFFN(8, 32, activation="silu")
    output_silu = model_silu(x)
    print(f"使用 SiLU 的 FFN 输出形状: {output_silu.shape}")

    model_gelu = SimpleFFN(8, 32, activation="gelu")
    output_gelu = model_gelu(x)
    print(f"使用 GELU 的 FFN 输出形状: {output_gelu.shape}")

    model_relu = SimpleFFN(8, 32, activation="relu")
    output_relu = model_relu(x)
    print(f"使用 ReLU 的 FFN 输出形状: {output_relu.shape}\n")


def example_list_available():
    """列出所有可用的激活函数"""
    print("=" * 60)
    print("所有可用的激活函数")
    print("=" * 60)

    from src.layers.activation import ACTIVATION_IMPLEMENTATIONS

    print("可用的激活函数:")
    for name in ACTIVATION_IMPLEMENTATIONS.keys():
        print(f"  - {name}")
    print()


def example_comparison():
    """比较不同激活函数的输出"""
    print("=" * 60)
    print("激活函数输出比较")
    print("=" * 60)

    x = torch.linspace(-2, 2, 5)
    print(f"输入: {x}\n")

    activations = ["relu", "silu", "gelu", "tanh", "sigmoid"]

    for act_name in activations:
        output = apply_activation(x, activation=act_name)
        print(f"{act_name:10s}: {output}")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_with_parameters()
    example_in_model()
    example_list_available()
    example_comparison()

    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
