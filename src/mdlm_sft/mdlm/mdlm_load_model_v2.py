import math
from pathlib import Path

from huggingface_hub import download_bucket_files
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

def resize_mdlm_vocab(model, new_vocab: int) -> None:
    backbone = model.backbone
    in_emb = backbone.vocab_embed.embedding  # nn.Parameter [V, H]
    out_lin = backbone.output_layer.linear  # nn.Linear(H, V)

    old_vocab, hidden = in_emb.shape
    assert out_lin.weight.shape == (old_vocab, hidden)
    assert out_lin.bias.shape == (old_vocab,)
    
    if new_vocab == old_vocab:
        return
    if new_vocab < old_vocab:
        raise ValueError(f"shrinking vocab not supported ({old_vocab} -> {new_vocab})")

    device_, dtype_ = in_emb.device, in_emb.dtype

    # --- input embedding ---------------------------------------------------
    new_in = torch.empty((new_vocab, hidden), device=device_, dtype=dtype_)
    torch.nn.init.kaiming_uniform_(new_in, a=math.sqrt(5))  # match EmbeddingLayer init
    with torch.no_grad():
        new_in[:old_vocab] = in_emb.data
    backbone.vocab_embed.embedding = torch.nn.Parameter(new_in)

    # --- output projection -------------------------------------------------
    new_w = torch.zeros((new_vocab, hidden), device=device_, dtype=out_lin.weight.dtype)
    new_b = torch.zeros((new_vocab,), device=device_, dtype=out_lin.bias.dtype)
    with torch.no_grad():
        new_w[:old_vocab] = out_lin.weight.data
        new_b[:old_vocab] = out_lin.bias.data
    out_lin.weight = torch.nn.Parameter(new_w)
    out_lin.bias = torch.nn.Parameter(new_b)
    out_lin.out_features = new_vocab
    model.config.vocab_size = new_vocab


def download_base_model() -> None:
    local_path = Path("artifacts_weights_mdlm_base_mdlm-owt")
    chat_path  = Path("artifacts_weights_mdlm_base_mdlm-owt_chat")

    required_files = [
        "model.safetensors",
        "modeling_mdlm.py",
        "config.json",
        "configuration_mdlm.py",
    ]
    local_path.mkdir(parents=True, exist_ok=True)
    files_to_download = [(f, str(local_path / f)) for f in required_files if not (local_path / f).exists()]

    if files_to_download:
        download_bucket_files("avgJo3/mdlm-owt-bucket", files=files_to_download)

    model     = AutoModelForMaskedLM.from_pretrained(str(local_path), trust_remote_code=True, device_map="auto")        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to(torch.bfloat16)


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

    tokenizer.chat_template = chat_template_str
    tokenizer.add_special_tokens(
        {
            "mask_token": "<mask>",  # gets id 50257 (first free slot)
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
            ],
        }
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token    


    old_vocab = model.backbone.vocab_embed.embedding.shape[0]
    print(f"[mdl] pretrained vocab size: {old_vocab}")
    
    assert old_vocab >= 50258, "checkpoint smaller than expected"
    mask_row_before = model.backbone.vocab_embed.embedding[50257].detach().clone().cpu()

    padded_vocab = math.ceil(len(tokenizer) / 64) * 64
    resize_mdlm_vocab(model, padded_vocab)
    new_vocab = model.backbone.vocab_embed.embedding.shape[0]
    model.save_pretrained(str(chat_path))
    tokenizer.save_pretrained(str(chat_path))

download_base_model()