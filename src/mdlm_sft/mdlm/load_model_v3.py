import gc
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def download_mdlm_cot_checkpoint() -> Path:
    """Download the MDLM-CoT checkpoint and materialize a flat, loadable model dir."""

    repo_id    = "avgJo3/mdlm_cot"
    checkpoint = "checkpoint-2554-split3"
    local_path = Path("artifacts_weights_mdlm_cot_s3")

    NEEDED = [
        "config.json",
        "configuration_mdlm.py",
        "modeling_mdlm.py",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]

    cache_dir = local_path.parent / f".hf_cache_{local_path.name}"
    snapshot_path = None
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{checkpoint}/{name}" for name in NEEDED],
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

        # --- Sanity check: every required file landed ---
        missing = [n for n in NEEDED if not (local_path / n).exists()]
        if missing:
            raise FileNotFoundError(f"missing files in {local_path}: {missing}")

        print(f"[ok] model dir ready at: {local_path.resolve()}")
        print(sorted(p.name for p in local_path.iterdir()))
        return local_path

    except Exception as e:
        print(f"[err] download_mdlm_cot_checkpoint failed: {type(e).__name__}: {e}")
        raise

    finally:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception as cleanup_err:
                print(f"[warn] cache cleanup failed: {cleanup_err}")
        snapshot_path = None
        gc.collect()


if __name__ == "__main__":
    download_mdlm_cot_checkpoint()