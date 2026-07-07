import tempfile
from pathlib import Path
import pprint
from time import time
from datasets import load_dataset
from huggingface_hub import download_bucket_files
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch 
import math

from mdlm_sft.mdlm.mdlm_gen_v3   import MDLMGenerationConfig, run_inference
from mdlm_sft.mdlm.mdlm_sft_v4   import MDLMSFTConfig, run_training


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


def download_base_model(target: str = "artifacts_weights_mdlm_base_mdlm-owt_chat") -> str:
    """Download MDLM base weights, extend tokenizer with chat/special tokens,
    resize the model's vocab embedding, and save the processed model+tokenizer.

    Args:
        target: Directory to write the processed (chat-ready) model + tokenizer to.
                Defaults to the project-local path for backward compatibility.

    Returns:
        The `target` path as a string, for convenient chaining.
    """
    local_path = Path("artifacts_weights_mdlm_base_mdlm-owt")   # raw download cache (project-local, unchanged)
    chat_path  = Path(target)                                    # processed output (parameterized)

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

    chat_path.mkdir(parents=True, exist_ok=True)                 # ← added: ensure target dir exists
    model.save_pretrained(str(chat_path))
    tokenizer.save_pretrained(str(chat_path))

    return str(chat_path)                                        # ← added: return path



# ── scratch dir ──────────────────────────────────────────────────────────────
SCRATCH = Path(tempfile.mkdtemp(prefix="mdlm-sft-"))
print(f"[stub] scratch dir: {SCRATCH}")

BASE_DATASET_PATH = "avgJo3/tinystories-strat"
DATA              = str(SCRATCH / "data")
MODEL             = str(SCRATCH / "model")


# ── round builders ───────────────────────────────────────────────────────────
def base_rounds(n_rounds: int) -> dict:
    rounds = {}
    for n in range(n_rounds):
        train_input = f"{DATA}_train" if n == 0 else f"{DATA}_train-generated_r{n-1}"
        rounds[f"R{n}-train"] = {
            "model_name_or_path": f"{MODEL}_base",
            "output_dir":         f"{MODEL}_trained_r{n}",
            "train_ds_path":      train_input,
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_trained_r{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_train-generated_r{n}",
        }
    return rounds


def ablation_rounds(n_rounds: int) -> dict:
    """ABLATION: reloads previous round's checkpoint (not base). Reuses BASE's R0."""
    rounds = {}
    for n in range(1, n_rounds + 1):
        prev_model = f"{MODEL}_trained_r0"        if n == 1 else f"{MODEL}_ablation-trained_r{n-1}"
        prev_data  = f"{DATA}_train-generated_r0" if n == 1 else f"{DATA}_ablation-generated_r{n-1}"
        rounds[f"R{n}-train"] = {
            "model_name_or_path": prev_model,
            "output_dir":         f"{MODEL}_ablation-trained_r{n}",
            "train_ds_path":      prev_data,
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_ablation-trained_r{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_ablation-generated_r{n}",
        }
    return rounds


# ── config ───────────────────────────────────────────────────────────────────
config = {
    "train_overrides": {
        "max_steps":                    4,
        "eval_steps":                   2,
        "logging_steps":                1,
        "per_device_train_batch_size":  2,
        "per_device_eval_batch_size":   2,
        "gradient_accumulation_steps":  1,
        "torch_compile":                False,
        "activation_offloading":        False,
        "bf16":                         True,
        "fp16":                         False,
        "report_to":                    "none",
        "eval_strategy":                "no",
        "eval_on_start":                False,
        "save_strategy":                "no",
    },
    "RUNS": {
        "BASE": {
            "train_overrides": {"learning_rate": 1e-5},
            "ROUNDS": base_rounds(n_rounds=1),      # ← start with 1 round, not 2
        },
    },
}


def upload_artifacts_to_bucket(scratch_dir: Path, *, namespace: str, bucket_name: str, private: bool = True) -> str:
    now = datetime.now(timezone.utc)
    digest = hashlib.sha256(now.isoformat().encode()).hexdigest()[:16]
    run_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{digest}"

    create_bucket(bucket_name, private=private, exist_ok=True)
    remote = f"hf://buckets/{namespace}/{bucket_name}/{run_id}"
    print(f"[upload] syncing {scratch_dir} → {remote}")
    sync_bucket(str(scratch_dir), remote)

    print(f"[upload] contents:")
    for item in list_bucket_tree(f"{namespace}/{bucket_name}", prefix=run_id, recursive=True):
        print(f"  {item.path}  ({item.size} bytes)")
    return remote



pprint.pprint(config, sort_dicts=False, width=120)
# ── prepare roots ────────────────────────────────────────────────────────────
download_base_model(target=f"{MODEL}_base")

dsd = load_dataset(BASE_DATASET_PATH)
print(f"[stub] dataset splits: {list(dsd.keys())}")           # ← verify split names before saving
dsd["train"].save_to_disk(f"{DATA}_train")
dsd["validation"].save_to_disk(f"{DATA}_validation")


# ── executor ─────────────────────────────────────────────────────────────────
global_ov = config["train_overrides"]
for run_name, run in config["RUNS"].items():
    run_ov = run.get("train_overrides", {})
    for stage, sc in run["ROUNDS"].items():
        print(f"\n=== {run_name} / {stage} ===")
        if stage.endswith("-train"):
            merged = {**global_ov, **run_ov}
            run_training(MDLMSFTConfig(**sc, **merged), save_last=True)
        elif stage.endswith("-inference"):
            run_inference(MDLMGenerationConfig(**sc))
        else:
            raise ValueError(f"unknown stage suffix: {stage!r}")
        
delay = 30
for attempt in range(1, 4):
    try:
        remote = upload_artifacts_to_bucket(
            SCRATCH,
            namespace="avgJo3",
            bucket_name="mdlm-sft-artifacts",
            private=True,
        )
        print(f"[done] artifacts uploaded to {remote}")
        break
    except Exception as e:
        print(f"[upload] attempt {attempt}/3 failed: {type(e).__name__}: {e}")
        if attempt == 3:
            print(f"[upload] giving up. Local artifacts at: {SCRATCH}")
            raise
        time.sleep(delay)
        delay *= 2        