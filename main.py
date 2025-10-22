import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================
# argument parsing
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B")
parser.add_argument("--draft-model", type=str, default=None)
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--solution", type=str, default="tinyllm")
parser.add_argument("--loader", type=str, default="v1")
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--sampler-temp", type=float, default=0)
parser.add_argument("--sampler-top-p", type=float, default=0)
parser.add_argument("--sampler-top-k", type=int, default=0)
parser.add_argument("--enable-thinking", action="store_true")
parser.add_argument("--enable-flash-attn", action="store_true")
args = parser.parse_args()

use_transformers = False
# =============================
# load model implementations
# =============================
if args.solution == "tinyllm":
    print("Using your tinyllm solution")
    from tests.tinyllm_base import (
        dispatch_model,
        shortcut_name_to_full_name,
        simple_generate,
        # simple_generate_with_kv_cache,
        # speculative_generate,
        make_sampler,
    )
elif args.solution == "transformers":
    print("Using transformers solution")
    use_transformers = True
else:
    raise ValueError(f"Solution {args.solution} not supported")


# =============================
# Load main model & tokenizer
# =============================
print(f"Loading model {args.model} ...")
args.model = shortcut_name_to_full_name(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
model.to(args.device)
model.eval()

# =============================
# Load optional draft model
# =============================
if args.draft_model:
    print(f"Loading draft model {args.draft_model} ...")
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=torch.float16
    )
    draft_model.to(args.device)
    draft_model.eval()
else:
    draft_model = None
    draft_tokenizer = None

# =============================
# Build prompt
# =============================
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": args.prompt},
]

# 如果模型 tokenizer 支持 chat 模板（如 Qwen / LLaMA）
if hasattr(tokenizer, "apply_chat_template"):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking if "enable-thinking" in args else False,
    )
else:
    # 普通 prompt 直接拼接
    prompt = args.prompt

# =============================
# 构造 sampler
# =============================
sampler_fn = make_sampler(
    args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
)

# =============================
# Choose generation logic
# =============================
if use_transformers:
    tinyllm_model = model
    outputs = tinyllm_model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids.to(args.device),
        do_sample=True,
        temperature=args.sampler_temp,
        top_p=args.sampler_top_p,
        top_k=args.sampler_top_k,
        max_new_tokens=128,
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text)
else:
    if args.loader == "v1":
        print(f"Using simple_generate for {args.model}")
        tinyllm_model = dispatch_model(args.model, model, version=1)
        output_text = simple_generate(
            tinyllm_model, tokenizer, prompt, sampler=sampler_fn
        )
        print(output_text)

    # elif args.loader == "week2":
    #     if draft_model is not None:
    #         print(f"Using speculative_generate with draft model {args.draft_model}")
    #         output_text = speculative_generate(
    #             draft_model, model, draft_tokenizer, tokenizer, prompt
    #         )
    #     else:
    #         print(f"Using simple_generate_with_kv_cache for {args.model}")
    #         output_text = simple_generate_with_kv_cache(model, tokenizer, prompt)

    #     print(output_text)

    else:
        raise ValueError(f"Loader {args.loader} not supported")
