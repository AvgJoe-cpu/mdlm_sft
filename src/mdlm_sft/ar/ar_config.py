from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from hydra.core.config_store import ConfigStore

from ..paths import AR_MODELS, DATASETS, DATASETS_DIR, DATASET_BASE_DIR


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    model_name: str = "pythia-70m"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    for_training: bool = True
    
    # Resolved paths (computed dynamically, not from config)
    hf_path: Optional[str] = field(default=None, init=False)
    base_path: Optional[Path] = field(default=None, init=False)
    checkpoints_path: Optional[Path] = field(default=None, init=False)
    tokenizer_cache_path: Optional[Path] = field(default=None, init=False)
    
    def __post_init__(self):
        """Resolve paths based on model_name"""
        if self.model_name not in AR_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' not found in AR_MODELS. "
                f"Available models: {list(AR_MODELS.keys())}"
            )
        
        model_info = AR_MODELS[self.model_name]
        self.hf_path = model_info["hf-path"]
        self.base_path = model_info["base_path"]
        self.checkpoints_path = model_info["checkpoints_path"]
        self.tokenizer_cache_path = self.base_path / "tokenizer"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    dataset_key: str = "wrp"
    num_samples: int = 5
    split: str = "train"  # Added: which split to load
    
    # Resolved paths (computed dynamically)
    data_load_path: Optional[Path] = field(default=None, init=False)  # Input (shared base)
    data_save_path: Optional[Path] = field(default=None, init=False)  # Output (model-specific gen)
    
    def resolve_load_path(self):
        """
        Resolve input dataset path (shared base dataset)
        Structure: datasets/base/{dataset_key}/{split}/
        """
        if self.dataset_key not in DATASETS:
            raise ValueError(
                f"Dataset '{self.dataset_key}' not found in DATASETS. "
                f"Available datasets: {list(DATASETS.keys())}"
            )
        
        # Input from shared base location with split
        self.data_load_path = DATASET_BASE_DIR / self.dataset_key / self.split
    
    def resolve_save_path(self, model_type: str, model_name: str, suffix: str):
        """
        Resolve output dataset path (model-specific generated)
        Structure: datasets/gen/{dataset_key}/{model_type}/{model_name}_{suffix}/
        """
        # Output to model-specific gen location
        self.data_save_path = (
            DATASETS_DIR / "gen" / self.dataset_key / model_type /
            f"{model_name}_{suffix}"
        )


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters and settings"""
    num_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 2e-5
    bf16: bool = True
    optim: str = "adamw_torch"
    use_liger_kernel: bool = False
    
    logging_steps: int = 10
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    report_to: str = "wandb"  # Changed from tensorboard to wandb
    
    assistant_only_loss: bool = True
    remove_unused_columns: bool = False
    push_to_hub: bool = False


@dataclass
class InferenceConfig:
    """Configuration for inference/generation parameters only"""
    # Generation parameters
    max_new_tokens: int = 256
    num_beams: int = 1
    do_sample: bool = True
    use_cache: bool = True
    temperature: float = 1.0
    num_return_sequences: int = 1
    batch_size: int = 8
    
    # Output
    output_suffix: str = "generated"  # Appended to dataset save path


@dataclass
class RunConfig:
    """Composed configuration for a complete training run"""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Run-specific settings
    checkpoint_name: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 42
    
    # Resolved paths (computed dynamically)
    model_load_path: Optional[str] = field(default=None, init=False)
    model_save_path: Optional[Path] = field(default=None, init=False)
    wandb_log_dir: Optional[Path] = field(default=None, init=False)  # Changed from tensorboard_log_dir
    
    def __post_init__(self):
        """Resolve training-specific paths"""
        model_info = AR_MODELS[self.model.model_name]
        
        # Resolve dataset load path (shared base)
        self.dataset.resolve_load_path()
        
        # Resolve model paths
        if self.checkpoint_name:
            self.model_load_path = str(
                model_info["checkpoints_path"] / self.checkpoint_name
            )
            save_name = self.run_name or f"{self.checkpoint_name}_continued"
        else:
            self.model_load_path = model_info["hf-path"]
            save_name = self.run_name or f"{self.dataset.dataset_key}_sft"
        
        self.model_save_path = model_info["checkpoints_path"] / save_name
        self.wandb_log_dir = self.model_save_path / "wandb_logs"  # Changed from tensorboard


@dataclass
class InferenceRunConfig:
    """Configuration for inference run - reuses model and dataset configs"""
    model: ModelConfig = field(default_factory=lambda: ModelConfig(for_training=False))
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Checkpoint management
    checkpoint_name: Optional[str] = None
    seed: int = 42
    
    # Resolved paths
    model_load_path: Optional[str] = field(default=None, init=False)
    data_save_path: Optional[Path] = field(default=None, init=False)  # For backwards compat
    
    def __post_init__(self):
        """Resolve inference-specific paths"""
        model_info = AR_MODELS[self.model.model_name]
        
        # Resolve dataset paths
        # Load from: datasets/base/wrp/train/
        self.dataset.resolve_load_path()
        
        # Save to: datasets/gen/wrp/ar/pythia-70m_test_generated/
        self.dataset.resolve_save_path(
            model_info["model_type"],
            self.model.model_name,
            self.inference.output_suffix
        )
        
        # Expose save path at top level for convenience
        self.data_save_path = self.dataset.data_save_path
        
        # Resolve model load path
        if self.checkpoint_name:
            self.model_load_path = str(
                model_info["checkpoints_path"] / self.checkpoint_name
            )
        else:
            self.model_load_path = model_info["hf-path"]


def register_configs() -> None:
    """Register configurations with Hydra ConfigStore"""
    cs = ConfigStore.instance()
    
    # Training configs
    cs.store(name="train_config", node=RunConfig)
    
    # Inference config
    cs.store(name="infer_config", node=InferenceRunConfig)
    
    # Shared model configs
    cs.store(group="model", name="pythia-70m", node=ModelConfig(model_name="pythia-70m"))
    cs.store(group="model", name="pythia-160m", node=ModelConfig(model_name="pythia-160m"))
    
    # Shared dataset configs
    cs.store(group="dataset", name="wrp", node=DatasetConfig(dataset_key="wrp"))
    cs.store(group="dataset", name="alp", node=DatasetConfig(dataset_key="alp"))
    cs.store(group="dataset", name="tis", node=DatasetConfig(dataset_key="tis"))