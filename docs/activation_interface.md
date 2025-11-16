# Activation 统一接口

## 概述

`src/layers/activation.py` 提供了一个统一的激活函数接口，类似于 `attention.py` 的设计模式。它允许通过字符串名称动态选择和应用不同的激活函数。

## 功能特性

✅ **统一接口**: 通过字符串名称选择激活函数  
✅ **类型安全**: 完整的类型注解  
✅ **易于扩展**: 简单添加新的激活函数实现  
✅ **参数支持**: 支持带参数的激活函数（如 Leaky ReLU）  
✅ **文档完善**: 详细的函数文档和使用示例  

## 支持的激活函数

| 名称 | 别名 | 描述 |
|------|------|------|
| `silu` | `swish` | SiLU/Swish激活函数: x * sigmoid(x) |
| `relu` | - | ReLU激活函数: max(0, x) |
| `gelu` | - | GELU激活函数: x * Φ(x) |
| `tanh` | - | Tanh激活函数 |
| `sigmoid` | - | Sigmoid激活函数 |
| `leaky_relu` | - | Leaky ReLU激活函数 |
| `mish` | - | Mish激活函数: x * tanh(softplus(x)) |
| `glu` | - | Gated Linear Unit |

## 使用方法

### 方法1: 获取激活函数

```python
from src.layers.activation import get_activation

# 获取激活函数
silu_fn = get_activation("silu")
output = silu_fn(input_tensor)

# 或者直接使用
output = get_activation("gelu")(input_tensor)
```

### 方法2: 使用统一接口

```python
from src.layers.activation import apply_activation

# 基本用法
output = apply_activation(input_tensor, activation="silu")

# 带参数的激活函数
output = apply_activation(
    input_tensor, 
    activation="leaky_relu", 
    negative_slope=0.1
)
```

### 方法3: 直接导入函数

```python
from src.layers.activation import silu, relu, gelu

output1 = silu(input_tensor)
output2 = relu(input_tensor)
output3 = gelu(input_tensor)
```

## 在模型中使用

### 示例1: 简单的前馈网络

```python
import torch.nn as nn
from src.layers.activation import apply_activation

class SimpleFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, activation="silu"):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.w1(x)
        x = apply_activation(x, activation=self.activation)
        x = self.w2(x)
        return x

# 使用不同的激活函数
model_silu = SimpleFeedForward(512, 2048, activation="silu")
model_gelu = SimpleFeedForward(512, 2048, activation="gelu")
model_relu = SimpleFeedForward(512, 2048, activation="relu")
```

### 示例2: 配置化模型

```python
from src.layers.activation import get_activation

class ConfigurableModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        # 从配置中获取激活函数
        self.activation_fn = get_activation(config.activation)
    
    def forward(self, x):
        return self.activation_fn(self.linear(x))

# 配置示例
class Config:
    hidden_size = 768
    activation = "gelu"  # 可以轻松切换: "silu", "relu", etc.

model = ConfigurableModel(Config())
```

### 示例3: Qwen2 MLP中的应用

```python
from src.layers import linear, get_activation

class Qwen2MLP(nn.Module):
    def __init__(self, dim, hidden_dim, w_gate, w_up, w_down, activation="silu"):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.register_buffer("w_gate", w_gate)
        self.register_buffer("w_up", w_up)
        self.register_buffer("w_down", w_down)
        self.activation_fn = get_activation(activation)

    def forward(self, x):
        # MLP(x) = (activation(W_gate(x)) ⊙ W_up(x)) W_down
        gate = self.activation_fn(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        return linear(gate * up, self.w_down)
```

## 添加新的激活函数

要添加新的激活函数，只需：

1. 在 `activation.py` 中实现函数：

```python
def my_activation(x: torch.Tensor) -> torch.Tensor:
    """My custom activation function."""
    return x * torch.exp(-x.abs())
```

2. 注册到字典中：

```python
ACTIVATION_IMPLEMENTATIONS: dict[str, Callable] = {
    # ... 其他激活函数
    "my_activation": my_activation,
}
```

3. 使用新激活函数：

```python
output = apply_activation(x, activation="my_activation")
```

## API 参考

### `get_activation(name: str) -> Callable`

根据名称获取激活函数。

**参数:**
- `name` (str): 激活函数名称

**返回:**
- `Callable`: 对应的激活函数

**异常:**
- `ValueError`: 如果提供的名称不存在

### `apply_activation(x: torch.Tensor, activation: str = "silu", **kwargs) -> torch.Tensor`

统一的激活函数应用接口。

**参数:**
- `x` (torch.Tensor): 输入张量
- `activation` (str): 激活函数名称，默认 "silu"
- `**kwargs`: 传递给激活函数的额外参数

**返回:**
- `torch.Tensor`: 应用激活函数后的张量

## 运行示例

```bash
# 运行使用示例
python examples/activation_usage.py

# 运行测试
pytest tests/test_activation.py -v
```

## 设计优势

1. **与 Attention 接口一致**: 采用相同的设计模式，便于学习和使用
2. **配置友好**: 可以通过配置文件字符串来选择激活函数
3. **易于实验**: 快速切换不同的激活函数进行实验
4. **向后兼容**: 仍然可以直接导入和使用单个激活函数
5. **类型安全**: 完整的类型注解，IDE友好

## 相关文件

- `src/layers/activation.py` - 激活函数实现和统一接口
- `examples/activation_usage.py` - 完整的使用示例
- `tests/test_activation.py` - 单元测试
