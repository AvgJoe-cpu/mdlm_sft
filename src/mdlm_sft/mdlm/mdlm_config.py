from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING            # <-- added

from ..paths import MDLM_MODELS, DATASETS, DATASET_BASE_DIR


# ============================================================ #
# Shared cores (used by BOTH training and inference)
# ============================================================ #
@dataclass
class ModelConfig:
    """Model + tokenizer location. Paths resolved from MDLM_MODELS."""
    model_name: str = MISSING          # was "mdlm-owt"
    tokenizer_name: str = MISSING      # was "gpt2"
    dtype: str = MISSING               # was "bfloat16"
    device_map: str = MISSING          # was "auto"

    # derived (NOT user inputs) -- unchanged
    hf_path: Optional[str] = field(default=None, init=False)
    base_path: Optional[Path] = field(default=None, init=False)
    checkpoints_path: Optional[Path] = field(default=None, init=False)
    tokenizer_cache_path: Optional[Path] = field(default=None, init=False)

    def __post_init__(self):
        if self.model_name not in MDLM_MODELS:
            raise ValueError(f"Unknown model '{self.model_name}'. Choices: {list(MDLM_MODELS)}")
        info = MDLM_MODELS[self.model_name]
        self.hf_path = info["hf-path"]
        self.base_path = info["base_path"]
        self.checkpoints_path = info["checkpoints_path"]
        self.tokenizer_cache_path = self.base_path / "tokenizer"


@dataclass
class DatasetConfig:
    dataset_key: str = MISSING             # was "wrp"
    num_train_samples: int = MISSING       # was 10000
    num_test_samples: int = MISSING        # was 1000
    max_length: int = MISSING              # was 1024

    # derived (NOT user inputs) -- unchanged
    train_data_load_path: Optional[Path] = field(default=None, init=False)
    test_data_load_path: Optional[Path] = field(default=None, init=False)

    def __post_init__(self):
        if self.dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset '{self.dataset_key}'. Choices: {list(DATASETS)}")
        base = DATASET_BASE_DIR / self.dataset_key
        self.train_data_load_path = base / "train"
        self.test_data_load_path = base / "test"


# ============================================================ #
# TRAINING
# ============================================================ #
@dataclass
class TrainingConfig:
    """Training hyperparameters (this is what a sweep varies)."""
    num_epochs: int = MISSING
    batch_size: int = MISSING
    learning_rate: float = MISSING
    warmup_ratio: float = MISSING
    weight_decay: float = MISSING
    grad_clip: float = MISSING

    num_workers: int = MISSING
    logging_steps: int = MISSING
    seed: int = MISSING

    adam_beta1: float = MISSING
    adam_beta2: float = MISSING

    # MDLM-specific
    scheduler: str = MISSING
    loss_weight_type: str = MISSING
    time_epsilon: float = MISSING

    # eval / save / reporting
    eval_strategy: str = MISSING
    eval_steps: int = MISSING
    save_strategy: str = MISSING
    report_to: str = MISSING    
    batch_eval_metrics: bool = MISSING
    remove_unused_columns: bool = MISSING
    bf16: bool = MISSING


@dataclass
class TrainConfig:
    """A complete training run."""
    # composition factories -- unchanged
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    checkpoint_name: Optional[str] = MISSING   # was None
    run_name: Optional[str] = MISSING          # was None
    seed: int = MISSING                        # was 42

    # derived (NOT user inputs) -- unchanged
    model_load_path: Optional[str] = field(default=None, init=False)
    model_save_path: Optional[Path] = field(default=None, init=False)
    tensorboard_log_dir: Optional[Path] = field(default=None, init=False)

    def __post_init__(self):
        ckpts = self.model.checkpoints_path
        if self.checkpoint_name:
            self.model_load_path = str(ckpts / self.checkpoint_name)
            save_name = self.run_name or f"{self.checkpoint_name}_continued"
        else:
            self.model_load_path = str(self.model.base_path)
            save_name = self.run_name or f"{self.dataset.dataset_key}_sft"
        self.model_save_path = ckpts / save_name
        self.tensorboard_log_dir = self.model_save_path / "logs"


# ============================================================ #
# INFERENCE  (left unchanged on purpose -- no YAML feeds these)
# ============================================================ #
@dataclass
class InferenceConfig:
    """MDLM sampling / generation parameters."""
    num_sample_steps: int = 128
    seq_len: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    num_samples: int = 16
    batch_size: int = 16
    seed: int = 42


@dataclass
class InferConfig:
    """A complete inference run."""
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    checkpoint_name: Optional[str] = None
    output_dir: Optional[str] = None

    model_load_path: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        self.model_load_path = str(
            self.model.checkpoints_path / self.checkpoint_name
            if self.checkpoint_name else self.model.base_path
        )


# ============================================================ #
# Registration
# ============================================================ #
def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="mdlm_train_config", node=TrainConfig)
    cs.store(name="mdlm_infer_config", node=InferConfig)

    cs.store(group="model", name="mdlm-owt", node=ModelConfig(model_name="mdlm-owt"))

    for key in ("wrp", "alp", "tis"):
        cs.store(group="dataset", name=key, node=DatasetConfig(dataset_key=key))