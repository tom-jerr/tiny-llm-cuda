import torch
from typing import Callable
from transformers import PreTrainedTokenizer

from .qwen2 import Qwen2Modelv1

# from .qwen2 import Qwen2Modelv2


def simple_generate(
    model: Qwen2Modelv1,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 128,
) -> str:
    """
    Simple autoregressive text generation using greedy decoding.

    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input text prompt
        sampler: Optional sampling function (default: argmax/greedy)
        device: Device to run on
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Generated text string untill EOS appeared
    """

    def _step(model, y):
        """
        Single generation step.

        Args:
            model: The language model
            y: Input token IDs of shape (N, S) where N=1 (batch size) and S=sequence length

        Returns:
            Next token ID of shape (N, 1)
        """
        # Forward pass through model: y (N, S) -> output_logits (N, S, vocab_size)
        output_logits = model(y)
        logits = output_logits[:, -1, :]  # (N, S, vocab_size) -> (N, vocab_size)
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Sample next token
        if sampler is not None:
            next_token = sampler(logits)
        else:
            next_token = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # Greedy decoding: keep (N, 1)
        return next_token

    # Setup
    model.eval().to(device)

    # Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    # Generate tokens autoregressively
    with torch.no_grad():
        while True:
            next_token = _step(model, generated)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# def simple_generate_with_kv_cache(
#     model: Qwen2ModelWeek2,
#     tokenizer: PreTrainedTokenizer,
#     prompt: str,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
# ) -> str:
#     model.eval().to(device)
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     generated = input_ids

#     with torch.no_grad():
#         # 第一次前向传播初始化 kv_cache
#         outputs = model(generated, use_cache=True)
#         kv_cache = outputs.past_key_values

#         for _ in range(128):
#             next_token_logits = outputs.logits[:, -1, :]
#             next_token = torch.argmax(next_token_logits, dim=-1)

#             generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

#             if next_token.item() == tokenizer.eos_token_id:
#                 break

#             # 增量生成，使用缓存
#             outputs = model(next_token.unsqueeze(-1), use_cache=True, past_key_values=kv_cache)
#             kv_cache = outputs.past_key_values

#     return tokenizer.decode(generated[0], skip_special_tokens=True)


# def speculative_generate(
#     draft_model: Qwen2ModelWeek2,
#     model: Qwen2ModelWeek2,
#     draft_tokenizer: PreTrainedTokenizer,
#     tokenizer: PreTrainedTokenizer,
#     prompt: str,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
# ) -> str:
#     """
#     简化版 speculative decoding: 先用 draft model 生成若干 token，
#     再用 main model 验证并采纳部分结果。
#     """
#     draft_model.eval().to(device)
#     model.eval().to(device)

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     generated = input_ids

#     with torch.no_grad():
#         for _ in range(128):
#             # draft model 预测若干 token
#             draft_outputs = draft_model(generated, use_cache=True)
#             draft_logits = draft_outputs.logits[:, -1, :]
#             draft_next = torch.argmax(draft_logits, dim=-1)

#             # main model 验证
#             main_outputs = model(generated, use_cache=True)
#             main_logits = main_outputs.logits[:, -1, :]

#             # 如果两个模型预测一致，就采纳
#             if draft_next.item() == torch.argmax(main_logits, dim=-1).item():
#                 next_token = draft_next
#             else:
#                 next_token = torch.argmax(main_logits, dim=-1)

#             generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

#             if next_token.item() == tokenizer.eos_token_id:
#                 break

#     return tokenizer.decode(generated[0], skip_special_tokens=True)
