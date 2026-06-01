from .ar_config import ModelConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

    
def load_model_and_tokenizer(cfg: ModelConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    chat_template_str = """
    {%- for message in messages %}
    {%- if message['role'] == 'system' %}
    {{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' %}
    {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
    {{- '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{- '<|im_end|>' }}
    {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- endif %}
    """.strip()

    # ── tokenizer: load from cache or download & save ─────────────────────
    if cfg.tokenizer_cache_path.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer_cache_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_path)
        tokenizer.chat_template = chat_template_str
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|user|>",
                    "<|assistant|>",
                    "<|system|>",
                ]
            }
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg.tokenizer_cache_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(cfg.tokenizer_cache_path))

    tokenizer.padding_side = "right" if cfg.for_training else "left"

    # ── model ─────────────────────────────────────────────────────────────
    dtype = getattr(torch, cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_path, dtype=dtype, device_map=cfg.device_map
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    return model, tokenizer