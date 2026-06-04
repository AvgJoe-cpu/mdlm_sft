from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import math
import torch
from huggingface_hub import download_bucket_files
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .mdlm_config import ModelConfig


def resize_mdlm_vocab(model, new_vocab: int) -> None:
    """Resize MDLM vocabulary embeddings to new size (append-only)."""
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


def _convert_to_bfloat16(model, dtype: str):
    """Convert model to bfloat16 when requested and supported."""
    if dtype in ("bfloat16", "auto"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
            print("[mdl] Converted to bfloat16 (CUDA)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to(torch.bfloat16)
            print("[mdl] Converted to bfloat16 (MPS)")
        else:
            print("[mdl] bfloat16 not supported, keeping float32")
    return model


def load_model_and_tokenizer(
    cfg: ModelConfig,
    load_path: Optional[str] = None,
    is_checkpoint: bool = False,
):
    """
    Load an MDLM model + tokenizer.

    Two modes:
      * is_checkpoint=True  -> load a self-contained trained checkpoint
        (weights + tokenizer with chat template / special tokens / resized
        vocab already baked in). No download, no resize. Used to warm-start
        a *fresh* run (new optimizer/scheduler/dataset) from a prior model.
      * is_checkpoint=False -> build the base model from the bucket: download
        artifacts, attach chat template, add ChatML specials, resize vocab.

    Args:
        cfg: ModelConfig with model_name, tokenizer_name, dtype.
        load_path: directory to load from when is_checkpoint=True (a
            self-contained HF checkpoint dir). Ignored otherwise.
        is_checkpoint: select the checkpoint fast path.

    Base-model order of operations (do NOT reorder):
      1. Download checkpoint artifacts from the bucket.
      2. Load model from the local snapshot.
      3. Build tokenizer -> attach chat template -> add special tokens
         -> set pad token. This finalizes len(tokenizer) BEFORE resize.
      4. Snapshot the MASK row at id 50257.
      5. Resize the model vocab to the next multiple of 64 >= len(tokenizer).
      6. Verify shapes and that the MASK row is byte-identical.
    """

    # ---- Fast path: self-contained trained checkpoint ---------------------
    # The checkpoint dir already has the resized vocab, ChatML special tokens
    # and chat template baked in, so we load weights + tokenizer directly and
    # skip the entire base pipeline (download / add specials / resize).
    if is_checkpoint:
        ckpt = Path(load_path)
        print(f"[mdl] Loading self-contained checkpoint from: {ckpt}")
        model = AutoModelForMaskedLM.from_pretrained(str(ckpt), trust_remote_code=True)
        print(f"[mdl] Model dtype: {next(model.parameters()).dtype}")
        model = _convert_to_bfloat16(model, cfg.dtype)

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[tok] Loaded checkpoint tokenizer (vocab size {len(tokenizer)})")
        print(f"[tok] mask_token : {tokenizer.mask_token!r}  (id {tokenizer.mask_token_id})")
        print(f"[tok] pad_token  : {tokenizer.pad_token!r}  (id {tokenizer.pad_token_id})")

        # Sanity: model vocab must cover the tokenizer.
        model_vocab = model.backbone.vocab_embed.embedding.shape[0]
        assert model_vocab >= len(tokenizer), (
            f"checkpoint model vocab {model_vocab} < tokenizer {len(tokenizer)}"
        )
        print("[mdl] Checkpoint model + tokenizer loaded successfully!")
        return model, tokenizer

    # ---- Base-model pipeline ----------------------------------------------
    # --- Chat template & special tokens ------------------------------------
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

    # Derived config paths arrive as strings (OmegaConf interpolation);
    # wrap the ones we do Path arithmetic on.
    base_path = Path(cfg.base_path)
    tokenizer_cache_path = Path(cfg.tokenizer_cache_path)

    # 1. ---- Ensure base path exists ----------------------------------------
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[mdl] Model repository: {cfg.hf_path}")
    print(f"[mdl] Cache directory: {base_path}")
    
    # 2. ---- Download model artifacts (only if missing) ---------------------
    required_files = [
        ("model.safetensors", base_path / "model.safetensors"),
        ("modeling_mdlm.py", base_path / "modeling_mdlm.py"),
        ("config.json", base_path / "config.json"),
        ("configuration_mdlm.py", base_path / "configuration_mdlm.py"),
    ]
    
    # Check which files need downloading
    files_to_download = []
    for remote_name, local_path in required_files:
        if not local_path.exists():
            files_to_download.append((remote_name, str(local_path)))
        else:
            print(f"[mdl] ✓ Cached: {local_path.name}")
    
    # Download missing files
    if files_to_download:
        print(f"[mdl] Downloading {len(files_to_download)} missing files...")
        download_bucket_files(cfg.hf_path, files=files_to_download)
    else:
        print("[mdl] ✓ All model files cached, skipping download")
    
    # 3. ---- Load model from local cache ------------------------------------
    print(f"[mdl] Loading model from: {base_path}")
    model = AutoModelForMaskedLM.from_pretrained(
        str(base_path),
        trust_remote_code=True,
    )
    
    print(f"[mdl] Model dtype: {next(model.parameters()).dtype}")
    
    # Convert to bfloat16 if requested
    model = _convert_to_bfloat16(model, cfg.dtype)
    
    # 4. ---- Tokenizer: base -> template -> specials -> pad ----------------
    if tokenizer_cache_path.is_dir():
        print(f"[tok] Loading cached tokenizer from: {tokenizer_cache_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_cache_path))
    else:
        print(f"[tok] Loading base tokenizer: {cfg.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        tokenizer_cache_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_cache_path))

    print(f"[tok] base tokenizer      : {cfg.tokenizer_name}")
    print(f"[tok] base vocab size     : {len(tokenizer)}")
    print(f"[tok] bos_token           : {tokenizer.bos_token!r}  (id {tokenizer.bos_token_id})")
    print(f"[tok] eos_token           : {tokenizer.eos_token!r}  (id {tokenizer.eos_token_id})")
    print(f"[tok] unk_token           : {tokenizer.unk_token!r}  (id {tokenizer.unk_token_id})")
    print(f"[tok] pad_token (before)  : {tokenizer.pad_token!r}  (id {tokenizer.pad_token_id})")
    print(f"[tok] mask_token (before) : {tokenizer.mask_token!r}  (id {tokenizer.mask_token_id})")
    base_special = tokenizer.all_special_tokens
    print(f"[tok] special tokens ({len(base_special)}) : {base_special}")

    # Attach chat template and add special tokens
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
                "<answer>",
                "<think>",
                "</think>",
            ],
        }
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[tok] vocab size after adding specials: {len(tokenizer)}")
    print(f"[tok] mask_token (after)  : {tokenizer.mask_token!r}  (id {tokenizer.mask_token_id})")
    print(f"[tok] pad_token  (after)  : {tokenizer.pad_token!r}  (id {tokenizer.pad_token_id})")
    all_special = tokenizer.all_special_tokens
    print(f"[tok] all special tokens ({len(all_special)}):")
    for tok in all_special:
        print(f"        {tok!r:25s}  id={tokenizer.convert_tokens_to_ids(tok)}")

    # 5. ---- Verify pretrained vocab and snapshot the MASK row -------------
    old_vocab = model.backbone.vocab_embed.embedding.shape[0]
    print(f"[mdl] pretrained vocab size: {old_vocab}")
    
    assert old_vocab >= 50258, "checkpoint smaller than expected"
    mask_row_before = model.backbone.vocab_embed.embedding[50257].detach().clone().cpu()

    # 6. ---- Grow model vocab to next multiple of 64 >= len(tokenizer) -----
    padded_vocab = math.ceil(len(tokenizer) / 64) * 64
    resize_mdlm_vocab(model, padded_vocab)

    # 7. ---- Post-conditions ------------------------------------------------
    new_vocab = model.backbone.vocab_embed.embedding.shape[0]
    print(f"[mdl] tokenizer vocab size : {len(tokenizer)}")
    print(f"[mdl] resized vocab size   : {new_vocab}  (padded to multiple of 64)")
    
    assert new_vocab == padded_vocab
    assert new_vocab >= len(tokenizer)
    assert model.backbone.output_layer.linear.weight.shape == (
        new_vocab,
        model.config.hidden_dim,
    )

    mask_row_after = model.backbone.vocab_embed.embedding[50257].detach().clone().cpu()
    assert torch.equal(
        mask_row_before, mask_row_after
    ), "MASK embedding row changed during resize — append-only invariant broken"

    print("[mdl] Model loaded and tokenizer configured successfully!")

    return model, tokenizer
