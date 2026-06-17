import math
from pathlib import Path

from huggingface_hub import download_bucket_files
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import gc
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


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


    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    chat_template_str = """
    {%- set default_system_prompt = __SYS__ %}
    {%- if messages[0]['role'] != 'system' %}
    {{- '<|im_start|>system\n' + default_system_prompt + '<|im_end|>\n' }}
    {%- endif %}
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
    """.strip().replace("__SYS__", repr(DEFAULT_SYSTEM_PROMPT))

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
                "<think>",
                "</think>",
                "<answer>",
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



def _download_one(checkpoint: str, local_path: Path, repo_id: str, needed: list[str]) -> Path:
    """Download a single checkpoint and flatten it into local_path/."""
    cache_dir = local_path.parent / f".hf_cache_{local_path.name}"
    snapshot_path = None
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{checkpoint}/{name}" for name in needed],
            local_dir=cache_dir,
        )
        src_dir = Path(snapshot_path) / checkpoint
        if not src_dir.is_dir():
            raise FileNotFoundError(
                f"expected checkpoint dir not found after download: {src_dir}"
            )
        local_path.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            shutil.copy2(f, local_path / f.name)

        missing = [n for n in needed if not (local_path / n).exists()]
        if missing:
            raise FileNotFoundError(f"missing files in {local_path}: {missing}")

        print(f"[ok] {checkpoint} -> {local_path.resolve()}")
        return local_path

    except Exception as e:
        print(f"[err] download {checkpoint} failed: {type(e).__name__}: {e}")
        raise

    finally:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception as cleanup_err:
                print(f"[warn] cache cleanup failed for {checkpoint}: {cleanup_err}")
        # Drop locals before the gc pass
        snapshot_path = None
        src_dir = None
        cache_dir = None
        gc.collect()


def download_mdlm_cot_checkpoint() -> dict[str, Path]:
    """Download all MDLM-CoT splits, each into its own flat local dir.

    All config is held *inside* this function — nothing leaks to module scope.
    Returns {checkpoint_name: local_path}.
    """
    repo_id = "avgJo3/mdlm_cot"
    needed = [
        "config.json",
        "configuration_mdlm.py",
        "modeling_mdlm.py",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    checkpoints = {
        "checkpoint-2554-split1": Path("artifacts_weights_mdlm_cot_s1"),
        "checkpoint-2554-split2": Path("artifacts_weights_mdlm_cot_s2"),
        "checkpoint-2554-split3": Path("artifacts_weights_mdlm_cot_s3"),
    }

    try:
        result = {
            ckpt: _download_one(ckpt, path, repo_id, needed)
            for ckpt, path in checkpoints.items()
        }
        return result
    finally:
        # Drop every local in this frame, then GC.
        del repo_id, needed, checkpoints
        try:
            del result   # may not exist if dict comp raised
        except NameError:
            pass
        gc.collect()

if __name__ == "__main__":
    download_base_model()
    download_mdlm_cot_checkpoint()