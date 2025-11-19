import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import (
    LMHead,
    dequantize_linear,
    qwen2,
)

from .utils import *


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("dtype", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_1_transformer_block(device: str, dtype: torch.dtype, mask: str | None):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    from transformers.models.qwen2 import Qwen2Config, modeling_qwen2

    BATCH_SIZE = 1
    SEQ_LEN = 10
    NUM_ATTENTION_HEAD = 4
    NUM_KV_HEADS = 2
    HIDDEN_SIZE = 32
    INTERMEDIATE_SIZE = HIDDEN_SIZE * 4

    config = Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEAD,
        num_key_value_heads=NUM_KV_HEADS,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        max_position_embeddings=1000,
    )
    config._attn_implementation = "sdpa"

    torch_transformer_block = (
        modeling_qwen2.Qwen2DecoderLayer(config, 0).to(device).to(dtype)
    )
    rotary_emb = modeling_qwen2.Qwen2RotaryEmbedding(config).to(device)

    torch_attention = torch_transformer_block.self_attn
    wq = torch_attention.q_proj.weight
    wk = torch_attention.k_proj.weight
    wv = torch_attention.v_proj.weight
    wo = torch_attention.o_proj.weight
    bq = torch_attention.q_proj.bias
    bk = torch_attention.k_proj.bias
    bv = torch_attention.v_proj.bias

    torch_mlp = torch_transformer_block.mlp
    w_gate = torch_mlp.gate_proj.weight
    w_up = torch_mlp.up_proj.weight
    w_down = torch_mlp.down_proj.weight

    w_input_layernorm = torch_transformer_block.input_layernorm.weight
    w_post_attention_layernorm = torch_transformer_block.post_attention_layernorm.weight

    user_config = qwen2.Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEAD,
        num_key_value_heads=NUM_KV_HEADS,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        max_position_embeddings=1000,
    )
    user_transformer_block = (
        qwen2.Qwen2TransformerBlock(
            user_config,
            w_input_layernorm=w_input_layernorm,
            w_post_attention_layernorm=w_post_attention_layernorm,
        )
        .to(device)
        .to(dtype)
    )

    # 复制权重到用户模型
    with torch.no_grad():
        user_transformer_block.self_attn.q_proj.weight.copy_(wq)
        user_transformer_block.self_attn.k_proj.weight.copy_(wk)
        user_transformer_block.self_attn.v_proj.weight.copy_(wv)
        user_transformer_block.self_attn.o_proj.weight.copy_(wo)
        user_transformer_block.self_attn.q_proj.bias.copy_(bq)
        user_transformer_block.self_attn.k_proj.bias.copy_(bk)
        user_transformer_block.self_attn.v_proj.bias.copy_(bv)
        user_transformer_block.mlp.gate_proj.weight.copy_(w_gate)
        user_transformer_block.mlp.up_proj.weight.copy_(w_up)
        user_transformer_block.mlp.down_proj.weight.copy_(w_down)

    torch.manual_seed(42)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device=device)
    position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(
        0
    )  # 需要 (bz, seq_len)
    position_embeddings = rotary_emb(x, position_ids)
    position_embeddings = tuple(
        pe.to(device=device, dtype=dtype) for pe in position_embeddings
    )

    with torch.no_grad():
        user_output = user_transformer_block(x, mask=mask)
        torch_output = torch_transformer_block(
            x,
            attention_mask=None,
            is_causal=(mask == "causal"),
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]

    assert_allclose(user_output, torch_output, precision=dtype, rtol=1e-1)


def helper_test_task_3(model_name: str, iters: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = qwen2.Qwen2Model(torch_model)

    with torch.no_grad():
        for _ in range(iters):
            input_ids = torch.randint(
                low=0, high=tokenizer.vocab_size, size=(1, 10), device=device
            )

            user_output = model(input_ids)
            user_output = user_output - torch.logsumexp(
                user_output, dim=-1, keepdim=True
            )

            ref_output = torch_model(input_ids).logits
            ref_output = ref_output - torch.logsumexp(ref_output, dim=-1, keepdim=True)

            assert_allclose(user_output, ref_output, precision=torch.float16, rtol=1e-1)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct model not found"
)
def test_task_2_embedding_call():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct", torch_dtype=torch.float16, device_map=device
    )

    embedding = LMHead(
        torch_model.config.vocab_size,
        torch_model.config.hidden_size,
        dequantize_linear(torch_model.model.embed_tokens).to(torch.float16),
    ).to(device)

    with torch.no_grad():
        for _ in range(50):
            input_ids = torch.randint(
                low=0, high=torch_model.config.vocab_size, size=(1, 10), device=device
            )

            user_output = embedding(input_ids)
            ref_output = torch_model.model.embed_tokens(input_ids)

            assert_allclose(user_output, ref_output, precision=torch.float16)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct model not found"
)
def test_task_2_embedding_as_linear():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct", dtype=torch.float16, device_map=device
    )

    lmhead = LMHead(
        torch_model.config.vocab_size,
        torch_model.config.hidden_size,
        dequantize_linear(torch_model.model.embed_tokens).to(torch.float16),
    ).to(device)

    with torch.no_grad():
        for _ in range(50):
            input_tensor = torch.randn(
                1,
                10,
                torch_model.config.hidden_size,
                dtype=torch.float16,
                device=device,
            )

            user_output = lmhead.as_linear(input_tensor)
            ref_output = torch.nn.functional.linear(
                input_tensor, torch_model.model.embed_tokens.weight
            )

            assert_allclose(user_output, ref_output, precision=torch.float16, atol=1e-1)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct model not found"
)
def test_task_3_qwen_2_05b():
    helper_test_task_3("Qwen/Qwen2-0.5B-Instruct", 5)


# @pytest.mark.skipif(
#     not qwen_2_7b_model_exists(), reason="Qwen2-7B-Instruct model not found"
# )
# def test_task_3_qwen_2_7b():
#     helper_test_task_3("Qwen/Qwen2-7B-Instruct", 1)


@pytest.mark.skipif(
    not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct model not found"
)
def test_task_3_qwen_2_15b():
    helper_test_task_3("Qwen/Qwen2-1.5B-Instruct", 3)
