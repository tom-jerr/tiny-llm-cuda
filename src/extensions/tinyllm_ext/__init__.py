import torch  # 先导入 torch 以加载必要的动态库
from ._ext import vector_add

__all__ = ["vector_add"]
