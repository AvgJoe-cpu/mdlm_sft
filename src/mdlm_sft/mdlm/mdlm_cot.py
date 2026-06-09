"""
Chain-of-Thought trainer for MDLM.

Key innovation: Teacher forcing at the reasoning-step level.
- Training data: question + previous_steps → current_step
- Only masks target tokens (current step), preserves context (question + previous steps)
- Enables iterative multi-step reasoning during inference

Based on "Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models"
(Ye et al., 2024) - https://arxiv.org/abs/2402.07754
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Any, Tuple, List
import math
import torch
import torch.nn.functional as F

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollator,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig

from mdlm_sft.mdlm.mdlm_sft_v2 import CustomForwardSFTTrainer
from mdlm_sft.mdlm.mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler


# ══════════════════════════════════════════════════════════════════════════════
# CoT Trainer (minimal override)
# ══════════════════════════════════════════════════════════════════════════════


class CoTSFTTrainer(CustomForwardSFTTrainer):
    """
    Chain-of-Thought trainer for MDLM.
    
    Extends CustomForwardSFTTrainer with step-level teacher forcing:
    - Respects `src_mask` to preserve question + previous reasoning steps
    - Only masks tokens in the current target reasoning step
    - All other behavior (metrics, eval, logging) inherited unchanged
    
    Data collator MUST provide:
        - input_ids: [question, SEP, prev_step_1, SEP, ..., current_step]
        - src_mask: 1 for question+prev_steps+SEP, 0 for current_step
        - labels: copy of input_ids (or -100 for padding)
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mode = "train" if self.model.training else "eval"

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        src_mask = inputs.get("src_mask", None)  # ← NEW: CoT-specific

        b, l = input_ids.shape

        # ── CHANGE: respect src_mask for CoT reasoning ────────────────────
        # Standard MDLM: maskable_mask = (labels != -100)
        # CoT-MDLM: maskable_mask = (labels != -100) & (~src_mask)
        #           → only mask target region (current reasoning step)
        if src_mask is not None:
            maskable_mask = (labels != -100) & (~src_mask)
        else:
            # Fallback to standard MDLM if src_mask not provided
            maskable_mask = labels != -100
        # ───────────────────────────────────────────────────────────────────

        n_maskable = maskable_mask.sum().clamp_min(1)

        # 1. timesteps (unchanged)
        t = self.time_epsilon + (1 - self.time_epsilon) * self._eval_rand(
            (b,), input_ids.device, mode, stream=0
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. masking (unchanged logic, operates on modified maskable_mask)
        masked_mask = (
            self._eval_rand((b, l), input_ids.device, mode, stream=1) < p_mask
        ) & maskable_mask

        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. forward (unchanged)
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)

        masked_logits = outputs.logits[masked_mask].float()
        masked_targets = input_ids[masked_mask]
        del outputs

        # 4. weights (unchanged)
        if self.loss_weight_type == "scheduler":
            loss_weights = (
                self.scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]
            )
        else:
            loss_weights = 1.0

        # 5. CE (unchanged)
        assert (input_ids[maskable_mask] == labels[maskable_mask]).all(), (
            "Mismatch between input_ids and labels at valid positions"
        )
        ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")

        # 6. loss (unchanged)
        local_loss = (ce * loss_weights).sum() / n_maskable
        if num_items_in_batch is not None:
            loss = local_loss * (n_maskable / num_items_in_batch)
        else:
            loss = local_loss

        # ── metrics (unchanged) ───────────────────────────────────────────
        with torch.no_grad():
            if mode == "train":
                if attention_mask is not None:
                    num_tokens_in_batch = (
                        self.accelerator.gather_for_metrics(attention_mask.sum())
                        .sum()
                        .item()
                    )
                else:
                    local_count = torch.tensor(l, device=input_ids.device)
                    num_tokens_in_batch = (
                        self.accelerator.gather_for_metrics(local_count).sum().item()
                    )
                self._total_train_tokens += num_tokens_in_batch
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

            log_probs = torch.log_softmax(masked_logits, dim=-1)
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            correct = masked_logits.argmax(dim=-1) == masked_targets
            del log_probs, masked_logits

            n_masked = masked_mask.sum()
            n_masked_g = self.accelerator.gather_for_metrics(n_masked).sum()
            correct_tokens = self.accelerator.gather_for_metrics(correct.sum()).sum()
            entropy_sum = (
                self.accelerator.gather_for_metrics(per_token_entropy.sum()).sum()
            )

            self._metric_sums[mode]["correct"] += correct_tokens.item()
            self._metric_sums[mode]["entropy"] += entropy_sum.item()
            self._metric_sums[mode]["tokens"] += n_masked_g.item()

            if mode == "eval":
                w = self.scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]
                batch_nll = (w.double() * ce.double()).sum()
                batch_toks = maskable_mask.sum()
                batch_nll = (
                    self.accelerator.gather_for_metrics(batch_nll).sum().item()
                )
                batch_toks = (
                    self.accelerator.gather_for_metrics(batch_toks).sum().item()
                )
                self._eval_nll_sum += batch_nll
                self._eval_token_sum += batch_toks
                self._eval_step += 1

        return (loss, {}) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════════════════
# CoT Data Collator
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoTDataCollator:
    """
    Collates Chain-of-Thought examples for step-level teacher forcing.
    
    Expected dataset format:
        {
            'question_tokens': List[int],              # question token IDs
            'reasoning_steps': List[List[int]],        # list of reasoning step token IDs
            'sep_token_id': int,                       # separator token
        }
    
    Generates multiple training examples per problem:
        - Example 0: question → step_0
        - Example 1: question + step_0 → step_1
        - Example 2: question + step_0 + step_1 → step_2
        - ...
    
    Output batch includes `src_mask` to mark context (question + prev steps).
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    expand_per_batch: bool = True  # If False, randomly sample one split per item

    def __call__(self, examples: List[dict]) -> dict[str, torch.Tensor]:
        """
        Collate examples with CoT reasoning structure.
        
        Args:
            examples: List of dicts with 'question_tokens', 'reasoning_steps', 'sep_token_id'
        
        Returns:
            Batch dict with input_ids, attention_mask, labels, src_mask
        """
        expanded = []
        
        for ex in examples:
            question = ex["question_tokens"]
            steps = ex["reasoning_steps"]
            sep_id = ex.get("sep_token_id", self.tokenizer.sep_token_id)
            
            # Determine how many training examples to create from this item
            if self.expand_per_batch:
                # Create one example per reasoning step (DoT-style expansion)
                splits = list(range(len(steps)))
            else:
                # Randomly sample one split (more predictable batch size)
                splits = [random.randint(0, len(steps) - 1)]
            
            for step_idx in splits:
                # Source: question + previous steps + SEP
                src = question.copy()
                for prev_idx in range(step_idx):
                    src.append(sep_id)
                    src.extend(steps[prev_idx])
                src.append(sep_id)
                
                # Target: current step
                tgt = steps[step_idx]
                
                # Full sequence
                full_seq = src + tgt
                src_len = len(src)
                
                # Truncate if needed
                if len(full_seq) > self.max_length:
                    # Try to keep target intact
                    tgt_len = len(tgt)
                    if tgt_len < self.max_length:
                        src_truncated = src[-(self.max_length - tgt_len):]
                        full_seq = src_truncated + tgt
                        src_len = len(src_truncated)
                    else:
                        # Target too long, truncate both
                        full_seq = full_seq[:self.max_length]
                        src_len = min(src_len, self.max_length)
                
                expanded.append({
                    "input_ids": full_seq,
                    "src_len": src_len,
                })
        
        # Convert to tensors and pad
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in expanded]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        # Create masks
        src_lens = torch.tensor([ex["src_len"] for ex in expanded], dtype=torch.long)
        seq_lens = torch.tensor([len(ex["input_ids"]) for ex in expanded], dtype=torch.long)
        
        # src_mask: 1 for question+prev_steps+SEP, 0 for current_step
        src_mask = (
            torch.arange(input_ids.shape[1])[None, :] < src_lens[:, None]
        )
        
        # attention_mask: 1 for real tokens, 0 for padding
        attention_mask = (
            torch.arange(input_ids.shape[1])[None, :] < seq_lens[:, None]
        )
        
        # labels: -100 for padding, input_ids elsewhere
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "src_mask": src_mask,  # NEW: marks context region
        }


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Processing
# ══════════════════════════════════════════════════════════════════════════════


def parse_gsm8k_reasoning(text: str) -> List[str]:
    """
    Parse GSM8K reasoning chain into steps.
    
    Format: "<<step1>> <<step2>> ... #### answer"
    Returns: ["<<step1>>", "<<step2>>", ..., "#### answer"]
    """
    # Split on spaces, but keep << >> blocks together
    parts = []
    current = []
    in_block = False
    
    for char in text:
        if char == '<' and len(current) > 0 and current[-1] == '<':
            in_block = True
        elif char == '>' and len(current) > 0 and current[-1] == '>':
            in_block = False
            current.append(char)
            parts.append(''.join(current))
            current = []
            continue
        elif char == ' ' and not in_block and current:
            part = ''.join(current).strip()
            if part and part not in ['<<', '>>']:
                parts.append(part)
            current = []
            continue
        
        current.append(char)
    
    if current:
        parts.append(''.join(current).strip())
    
    # Group into reasoning steps
    steps = []
    for part in parts:
        if part.startswith('<<') or part.startswith('####'):
            steps.append(part)
    
    return steps if steps else [text]  # Fallback to full text if parsing fails


def format_cot_example(
    example: dict, tokenizer: PreTrainedTokenizerBase
) -> dict:
    """
    Convert raw example to CoT format with tokenized reasoning steps.
    
    Expected input:
        {
            'prompt': "Question text",
            'completion': "<<step1>> <<step2>> #### answer"
        }
    
    Output:
        {
            'question_tokens': List[int],
            'reasoning_steps': List[List[int]],
            'sep_token_id': int
        }
    """
    question = example["prompt"]
    reasoning_text = example["completion"]
    
    # Parse reasoning into steps
    reasoning_steps_text = parse_gsm8k_reasoning(reasoning_text)
    
    # Tokenize
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    reasoning_steps = [
        tokenizer.encode(step, add_special_tokens=False)
        for step in reasoning_steps_text
    ]
    
    return {
        "question_tokens": question_tokens,
        "reasoning_steps": reasoning_steps,
        "sep_token_id": tokenizer.sep_token_id or tokenizer.eos_token_id,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training Config
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoTTrainingConfig:
    """Training configuration for CoT-MDLM."""
    
    # ── Experiment identity ───────────────────────────────────────────────
    output_dir: str = "output/cot-mdlm"
    model_name_or_path: str = "bert-base-uncased"
    run_name: Optional[str] = "cot-mdlm"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    # ── Data ──────────────────────────────────────────────────────────────
    max_length: int = 1024
    shuffle_dataset: bool = True
    dataset_num_proc: Optional[int] = None
    train_ds_path: str = "train_ds"
    eval_ds_path: str = "eval_ds"
    expand_per_batch: bool = True  # NEW: expand each problem into multiple examples

    # ── Optimization ──────────────────────────────────────────────────────
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: Optional[float] = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # ── Training loop ─────────────────────────────────────────────────────
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 5.0
    max_steps: int = -1
    bf16: bool = True

    # ── Eval ──────────────────────────────────────────────────────────────
    per_device_eval_batch_size: int = 1
    eval_strategy: str = "steps"
    eval_steps: int = 100
    eval_on_start: bool = True
    metric_for_best_model: str = "eval_nll"
    greater_is_better: bool = False
    load_best_model_at_end: bool = False

    # ── Logging ───────────────────────────────────────────────────────────
    logging_steps: int = 25
    report_to: str = "wandb"
    project: Optional[str] = "CoT-MDLM"

    # ── Memory & performance ──────────────────────────────────────────────
    gradient_checkpointing: bool = False
    activation_offloading: bool = True
    torch_compile: bool = True
    use_liger_kernel: bool = True
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 8
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True


cs = ConfigStore.instance()
cs.store(name="cot_config", node=CoTTrainingConfig)


# ══════════════════════════════════════════════════════════════════════════════
# Training Runner
# ══════════════════════════════════════════════════════════════════════════════


def run_cot_training(cfg: CoTTrainingConfig) -> None:
    """Main training loop for CoT-MDLM."""
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_name_or_path = cfg_dict.pop("model_name_or_path")
    resume_from_checkpoint = cfg_dict.pop("resume_from_checkpoint", None)
    expand_per_batch = cfg_dict.pop("expand_per_batch", True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    # Load and process datasets
    train_ds_path = cfg_dict.pop("train_ds_path")
    eval_ds_path = cfg_dict.pop("eval_ds_path")
    
    train_ds = load_from_disk(train_ds_path, keep_in_memory=True)
    eval_ds = load_from_disk(eval_ds_path, keep_in_memory=True)

    # Convert to CoT format
    train_ds = train_ds.map(
        format_cot_example,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Formatting train CoT examples",
    )
    eval_ds = eval_ds.map(
        format_cot_example,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Formatting eval CoT examples",
    )

    # Create CoT-specific data collator
    data_collator = CoTDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        expand_per_batch=expand_per_batch,
    )

    # Setup scheduler and training args
    scheduler = LinearAlphaScheduler()
    args = SFTConfig(
        **cfg_dict,
        logging_first_step=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Create CoT trainer
    trainer = CoTSFTTrainer(
        scheduler=scheduler,
        time_epsilon=1e-3,
        loss_weight_type="scheduler",
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,  # NEW: CoT-specific collator
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        # Cleanup
        del trainer, model, tokenizer, scheduler, args
        del train_ds, eval_ds, data_collator
        del cfg_dict, resume_from_checkpoint

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra.main(version_base=None, config_name="cot_config")
def main(cfg: CoTTrainingConfig) -> None:
    """Hydra entry point."""
    print("═" * 80)
    print("Chain-of-Thought MDLM Training")
    print("═" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("═" * 80)
    run_cot_training(cfg)


if __name__ == "__main__":
    main()