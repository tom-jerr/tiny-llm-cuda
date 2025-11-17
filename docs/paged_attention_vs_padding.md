# Batching Cache 的 Padding 问题与 PagedAttention 解决方案

## 当前实现的问题

### 为什么需要 Padding?

在批处理推理中,不同请求的序列长度不同:

```
Request 0: [tok_0, tok_1, ..., tok_49]      # 50 tokens
Request 1: [tok_0, tok_1, ..., tok_149]     # 150 tokens
Request 2: [tok_0, tok_1, ..., tok_79]      # 80 tokens
```

但 PyTorch 张量必须是规则形状,所以需要 padding 到统一长度:

```python
# 必须 padding 到 max_seq_len = 150
batched_keys.shape = (3, num_heads, 150, head_dim)

# 实际分布:
# Request 0: [0, 0, ..., 0, k_0, k_1, ..., k_49]    # 100个padding + 50个实际
# Request 1: [k_0, k_1, ..., k_149]                 # 0个padding + 150个实际
# Request 2: [0, 0, ..., 0, k_0, k_1, ..., k_79]    # 70个padding + 80个实际
```

### Padding 的代价

1. **内存浪费严重**
   - 批次大小 = 4, max_seq_len = 2048, 但平均长度只有 200
   - 实际使用: 4 × 200 = 800 tokens
   - 分配空间: 4 × 2048 = 8192 tokens
   - 浪费率: **90%**

2. **计算资源浪费**

   ```python
   # Attention 计算时虽然 mask 掉了 padding,但仍然:
   # - 读取 padding 的内存
   # - 计算 Q×K (padding部分)
   # - 执行条件判断 (mask)
   ```

3. **碎片化问题**
   - 短请求结束后,其占用的长 padding 空间无法复用
   - 长请求可能因为 batch 中其他短请求而被限制长度

---

## PagedAttention 解决方案

### 核心思想

借鉴**操作系统虚拟内存**的分页机制:

- 将 KV cache 分成固定大小的 **blocks**(如 16 tokens/block)
- 物理内存池中存储所有 blocks
- 每个请求维护一个**逻辑到物理的映射表**

### 架构对比

**传统 Padding 方式:**

```
物理内存布局(连续分配):
┌─────────────────────────────────────────┐
│ Req0: [tokens...] [padding...........] │  2048 tokens
├─────────────────────────────────────────┤
│ Req1: [tokens................] [pad..] │  2048 tokens
├─────────────────────────────────────────┤
│ Req2: [tokens.......] [padding.......] │  2048 tokens
└─────────────────────────────────────────┘
总共: 3 × 2048 = 6144 tokens (大量浪费)
```

**PagedAttention 方式:**

```
物理内存池(非连续,按需分配):
┌──────┬──────┬──────┬──────┬──────┬──────┐
│Block0│Block1│Block2│Block3│Block4│Block5│ ...
└──────┴──────┴──────┴──────┴──────┴──────┘
  ↑      ↑              ↑      ↑
  │      │              │      └─ Req2 使用
  │      │              └──────── Req1 使用
  │      └────────────────────── Req1 使用
  └───────────────────────────── Req0 使用

逻辑视图:
Req0 (50 tokens):  需要 4 blocks → [Block0]
Req1 (150 tokens): 需要 10 blocks → [Block1, Block3]
Req2 (80 tokens):  需要 5 blocks → [Block4]

总共只需: 19 blocks × 16 = 304 tokens (vs 6144)
```

### 实现原理

1. **Block 分配**

   ```python
   class BlockTable:
       def __init__(self, block_size=16):
           self.block_size = 16  # 每个block存16个token
           # 物理内存: [num_blocks, num_heads, block_size, head_dim]
           self.physical_blocks = allocate_kv_pool()
           self.free_blocks = list(range(num_blocks))

       def allocate_blocks(self, num_tokens):
           num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
           allocated = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
           return allocated  # 返回物理block ID列表
   ```

2. **逻辑到物理映射**

   ```python
   # 每个请求维护自己的 block table
   request_block_tables = {
       0: [7, 12],        # Req0: 逻辑block[0,1] → 物理block[7,12]
       1: [3, 8, 15],     # Req1: 逻辑block[0,1,2] → 物理block[3,8,15]
       2: [1, 9],         # Req2: 逻辑block[0,1] → 物理block[1,9]
   }
   ```

3. **自定义 Attention Kernel**

   ```cuda
   __global__ void paged_attention_kernel(
       Query* q,                    // [batch, heads, 1, dim]
       KVCache* kv_pool,           // 物理block池
       int* block_tables,          // [batch, max_blocks] 映射表
       int* seq_lengths,           // 每个请求的实际长度
       Output* out
   ) {
       int req_id = blockIdx.x;
       int head_id = blockIdx.y;
       int num_blocks = (seq_lengths[req_id] + BLOCK_SIZE - 1) / BLOCK_SIZE;

       float sum = 0;
       // 遍历此请求的所有逻辑blocks
       for (int logic_block = 0; logic_block < num_blocks; logic_block++) {
           int physical_block = block_tables[req_id * MAX_BLOCKS + logic_block];

           // 从物理内存池读取这个block的KV
           KVCache* kv_block = &kv_pool[physical_block];

           // 计算 attention (只针对实际数据,无padding!)
           for (int i = 0; i < BLOCK_SIZE; i++) {
               int global_pos = logic_block * BLOCK_SIZE + i;
               if (global_pos < seq_lengths[req_id]) {
                   sum += dot(q, kv_block[i]);
               }
           }
       }
       out[req_id] = sum;
   }
   ```

### 关键优势

1. **零内存浪费**
   - 只分配实际需要的 blocks
   - 平均长度 200 tokens → 13 blocks
   - 不再因为批次中最长请求而浪费

2. **动态共享**

   ```python
   # 多个请求可以共享相同的 prompt blocks
   # (Copy-on-Write 机制)
   system_prompt_blocks = [0, 1, 2]  # 共享的系统提示

   request_A.blocks = [0, 1, 2, 10, 11]  # 共享前3个,自己的2个
   request_B.blocks = [0, 1, 2, 12, 13]  # 共享前3个,自己的2个
   ```

3. **高效内存复用**
   - 请求结束 → 立即释放 blocks 到空闲池
   - 新请求可以立即使用释放的 blocks
   - 无碎片问题

---

## 是否还需要 Padding?

### PagedAttention 完全消除了 Padding!

**原因:**

1. **物理存储**: 每个 block 大小固定,无需 padding
2. **逻辑访问**: 通过 block table 映射,只访问实际使用的 blocks
3. **Attention 计算**: 自定义 kernel 知道每个请求的真实长度,只计算有效部分

**对比:**

```python
# 传统方式 - 需要 padding 到 max_seq_len
def batched_attention(Q, K, V, seq_lens):
    max_len = max(seq_lens)
    K_padded = pad_to_max_len(K, max_len)  # ❌ 必须 padding
    V_padded = pad_to_max_len(V, max_len)
    attention = torch.matmul(Q, K_padded.transpose(-2, -1))
    return apply_mask_and_softmax(attention, seq_lens)

# PagedAttention - 无需 padding
def paged_attention(Q, block_tables, kv_pool, seq_lens):
    # ✅ 直接通过 block_table 访问物理内存
    return paged_attention_kernel(Q, block_tables, kv_pool, seq_lens)
```

---

## 实现 PagedAttention 的步骤

如果要在这个项目中实现 PagedAttention:

### 1. 实现 Block Manager

```python
# src/cache/block_manager.py
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))

    def allocate(self, num_tokens: int) -> list[int]:
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return [self.free_blocks.pop() for _ in range(num_blocks)]

    def free(self, block_ids: list[int]):
        self.free_blocks.extend(block_ids)
```

### 2. 实现 PagedKVCache

```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        # 物理内存池: [num_blocks, num_heads, block_size, head_dim]
        self.kv_pool = torch.zeros((num_blocks, num_heads, block_size, head_dim))
        self.block_manager = BlockManager(num_blocks, block_size)
        self.block_tables = {}  # request_id -> list[block_id]

    def add_request(self, request_id, num_tokens):
        blocks = self.block_manager.allocate(num_tokens)
        self.block_tables[request_id] = blocks
```

### 3. 实现 CUDA Kernel (关键!)

```cuda
// src/extensions/ops/paged_attention.cu
__global__ void paged_attention_kernel(...) {
    // 根据 block_table 从 kv_pool 读取数据
    // 无需处理 padding
}
```

### 4. 替换现有的 BatchingKvCache

```python
# 不再需要:
batched_keys = torch.zeros((B, H, max_seq_len, D))  # ❌ padding

# 改用:
output = paged_attention(Q, block_tables, kv_pool, seq_lens)  # ✅ 无padding
```

---

## 参考资源

- [vLLM: PagedAttention 实现](https://github.com/vllm-project/vllm)
- [FlashInfer: 高性能 PagedAttention kernel](https://github.com/flashinfer-ai/flashinfer)
- 论文: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)

---

## 总结

| 特性         | 当前 Batching (Padding) | PagedAttention    |
| ------------ | ----------------------- | ----------------- |
| 内存利用率   | 10-40%                  | 90%+              |
| 需要 Padding | ✅ 是                   | ❌ 否             |
| 碎片化       | 严重                    | 无                |
| 共享 Prompt  | 不支持                  | 支持 (CoW)        |
| 实现复杂度   | 简单                    | 需要自定义 kernel |

**结论**: 如果实现了 PagedAttention,就**完全不需要 padding** 了!
