import torch
import copy
from .basics import softmax


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: torch.Tensor):
        """这里实现三种采样方式：温度采样、top-k采样、top-p采样
        Tempature sampling: 当 temp=0 时，我们使用默认的贪婪策略。当它大于 0 时，我们将根据对数概率随机选择下一个标记。温度参数会缩放分布。当该值较大时，分布将更加均匀，使得概率较低的标记更有可能被选中，从而使模型更具创造性。

        top-k sampling: 这种方法限制了我们在每一步采样时可以选择的标记数量。具体来说，我们只考虑具有最高对数概率的 k 个标记，并从中进行采样。这有助于防止模型选择非常不太可能的标记，从而提高生成文本的质量。

        top-p sampling: 这种方法动态地选择一个标记子集，其累积对数概率达到预定义的阈值 p。与 top-k 采样不同，top-p 采样根据当前分布调整其选择的标记数量。这允许模型在保持多样性的同时专注于更有可能的标记，从而生成更连贯和相关的文本。
        """
        if temp == 0:
            return torch.argmax(logprobs, axis=-1, keepdim=True)
        else:
            logprobs = logprobs / temp
            if top_p is not None and 0.0 < top_p < 1.0:
                # 1. sort the logprobs from largest to smallest
                sorted_logprobs, sorted_indices = torch.sort(
                    logprobs, descending=True, dim=-1
                )
                # 2. compute cumulative probabilities
                cumulative_probs = torch.cumsum(
                    softmax(sorted_logprobs, axis=-1), dim=-1
                )
                # 3. truncate tokens with cumulative prob above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep the first token above the threshold
                if sorted_indices_to_remove[..., 1:].any():
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = (
                        False  # like [False, False, True(keep last), True, True...]
                    )

                indices_to_remove = torch.zeros_like(
                    logprobs, dtype=torch.bool
                ).scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove,
                )  # result[index[i]] = src[i]

                logprobs = torch.where(
                    indices_to_remove,
                    torch.full_like(logprobs, float("-inf")),
                    logprobs,
                )
            elif top_k is not None and top_k > 0:
                topk_logprobs, _ = torch.topk(logprobs, k=top_k, dim=-1)
                min_topk_logprob = topk_logprobs[
                    ..., -1, None
                ]  # 取 seq 维度的最后一个值(即最小值)
                logprobs = torch.where(
                    logprobs < min_topk_logprob,
                    torch.full_like(logprobs, float("-inf")),
                    logprobs,
                )
            probs = softmax(logprobs, axis=-1)
            idxs = torch.multinomial(probs, num_samples=1)

            return idxs

    return sample
