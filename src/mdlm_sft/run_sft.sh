
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/content/mdlm_sft/artifacts_weights_mdlm_base_mdlm-owt_chat"

echo "=== Loading base model ==="
uv run python -m mdlm_sft.mdlm.mdlm_load_model_v2

# ---- Step 2: Download dataset from the hub & cache to disk -------------------
echo "=== Caching dataset to /content/writingprompts-strat ==="
uv run python - <<'EOF'
from datasets import load_dataset
dd = load_dataset("avgJo3/writingprompts-strat")
dd.save_to_disk("/content/writingprompts-strat")
print(dd)
EOF

# ---- Step 3: SFT runs --------------------------------------------------------
run_sft() {
    local train_path="$1"
    local eval_path="$2"
    local out_path="$3"

    echo "=== ${out_path} ==="
    mkdir -p "${out_path}"
    uv run python -m mdlm_sft.mdlm.mdlm_sft_v2 \
        model_name_or_path="${MODEL_PATH}" \
        train_ds_path="${train_path}" \
        eval_ds_path="${eval_path}" \
        output_dir="${out_path}" \
        per_device_train_batch_size=64 \
        gradient_accumulation_steps=10 \
        eval_steps=250 \
        per_device_eval_batch_size=64 \
        activation_offloading=false \
        lr_scheduler_type=cosine \
        torch_compile=True \
        warmup_ratio=0.05 \
        weight_decay=0.1 \
        learning_rate=5e-4 \
        adam_beta1=0.88 \
        adam_beta2=0.98 \
        max_grad_norm=2.0 \
        dataloader_num_workers=12 \
        dataloader_prefetch_factor=12 \
        max_steps=303 \
        num_train_epochs=9999
}

run_sft "/content/writingprompts-strat/strat/strat_train_12pct"  "/content/writingprompts-strat/strat_eval" "/content/wrp-strat-outdir_012"
run_sft "/content/writingprompts-strat/strat/strat_train_25pct"  "/content/writingprompts-strat/strat_eval" "/content/wrp-strat-outdir_025"
run_sft "/content/writingprompts-strat/strat/strat_train_50pct"  "/content/writingprompts-strat/strat_eval" "/content/wrp-strat-outdir_050"
run_sft "/content/writingprompts-strat/strat/strat_train_100pct" "/content/writingprompts-strat/strat_eval" "/content/wrp-strat-outdir_100"

echo ""
echo "✓ All runs complete."