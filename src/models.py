from .qwen2 import Qwen2ModelV1, Qwen2ModelV2


def shortcut_name_to_full_name(shortcut_name: str):
    lower_shortcut_name = shortcut_name.lower()
    if lower_shortcut_name == "qwen2-7b":
        return "Qwen/Qwen2-7B-Instruct"
    elif lower_shortcut_name == "qwen2-0.5b":
        return "Qwen/Qwen2-0.5B-Instruct"
    elif lower_shortcut_name == "qwen2-1.5b":
        return "Qwen/Qwen2-1.5B-Instruct"
    elif lower_shortcut_name == "qwen3-8b":
        return "mlx-community/Qwen3-8B-4bit"
    elif lower_shortcut_name == "qwen3-0.6b":
        return "mlx-community/Qwen3-0.6B-4bit"
    elif lower_shortcut_name == "qwen3-1.7b":
        return "mlx-community/Qwen3-1.7B-4bit"
    elif lower_shortcut_name == "qwen3-4b":
        return "mlx-community/Qwen3-4B-4bit"
    else:
        return shortcut_name


def dispatch_model(model_name: str, torch_model, version: int, **kwargs):
    model_name = shortcut_name_to_full_name(model_name)
    if version == 1 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelV1(torch_model, **kwargs)
    elif version == 2 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelV2(torch_model, **kwargs)
    # elif week == 2 and model_name.startswith("mlx-community/Qwen3"):
    #     return Qwen3Model(torch_model, **kwargs)
    else:
        raise ValueError(f"{model_name} for version {version} not supported")
