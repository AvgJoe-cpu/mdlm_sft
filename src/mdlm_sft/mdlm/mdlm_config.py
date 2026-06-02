from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from ..paths import MDLM_MODELS, DATASET_BASE_DIR, DATASET_GEN_DIR


# ============================================================ #
# Custom OmegaConf resolvers (the ONLY place paths are derived)
# ============================================================ #
def _model_field(model_name: str, key: str) -> str:
    return str(MDLM_MODELS[model_name][key])


def _ds_base(dataset_key: str, split: str) -> str:
    return str(DATASET_BASE_DIR / dataset_key / split)


def _ds_gen(model_type: str, dataset_key: str, run_name: str, save_name: str) -> str:
    # Mirrors mdlm_gen.py's mdlm_gen_out resolver so training can read exactly
    # what inference wrote:
    #   artifacts/datasets/gen/<model_type>/<dataset>/<run>/<save_name>
    return str(DATASET_GEN_DIR / model_type / dataset_key / run_name / save_name)


def _model_load_path(model_name: str, checkpoint_name: Optional[str]) -> str:
    info = MDLM_MODELS[model_name]
    if checkpoint_name:
        return str(info["checkpoints_path"] / checkpoint_name)
    return str(info["base_path"])


def _model_save_path(
    model_name: str,
    checkpoint_name: Optional[str],
    run_name: Optional[str],
    dataset_key: str,
) -> str:
    info = MDLM_MODELS[model_name]
    if run_name:
        save_name = run_name
    elif checkpoint_name:
        save_name = f"{checkpoint_name}_continued"
    else:
        save_name = f"{dataset_key}_sft"
    return str(info["checkpoints_path"] / save_name)


OmegaConf.register_new_resolver("mdlm_model", _model_field, replace=True)
OmegaConf.register_new_resolver("ds_base", _ds_base, replace=True)
OmegaConf.register_new_resolver("ds_gen", _ds_gen, replace=True)
OmegaConf.register_new_resolver("mdlm_load_path", _model_load_path, replace=True)
OmegaConf.register_new_resolver("mdlm_save_path", _model_save_path, replace=True)


# ============================================================ #
# Schema (type-checker only -- input defaults come from YAML,
#          derived paths come from resolvers above)
# ============================================================ #
@dataclass
class ModelConfig:
    model_name: str = MISSING
    tokenizer_name: str = MISSING
    dtype: str = MISSING
    device_map: str = MISSING

    # derived (resolved from MDLM_MODELS via interpolation)
    hf_path: str = "${mdlm_model:${.model_name},hf-path}"
    base_path: str = "${mdlm_model:${.model_name},base_path}"
    checkpoints_path: str = "${mdlm_model:${.model_name},checkpoints_path}"
    tokenizer_cache_path: str = "${mdlm_model:${.model_name},base_path}/tokenizer"


@dataclass
class DatasetConfig:
    dataset_key: str = MISSING
    num_train_samples: int = MISSING
    num_test_samples: int = MISSING
    max_length: int = MISSING

    # Split load paths. Default to the base tree, but each split can be
    # overridden in YAML to point at a generated dataset via ${ds_gen:...}.
    #   base : ${ds_base:<dataset_key>,<split>}
    #   gen  : ${ds_gen:<model_type>,<dataset_key>,<run_name>,<save_name>}
    train_data_load_path: str = "${ds_base:${.dataset_key},train}"
    test_data_load_path: str = "${ds_base:${.dataset_key},test}"


@dataclass
class TrainingConfig:
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
    scheduler: str = MISSING
    loss_weight_type: str = MISSING
    time_epsilon: float = MISSING
    eval_strategy: str = MISSING
    eval_steps: int = MISSING
    save_strategy: str = MISSING
    report_to: str = MISSING
    batch_eval_metrics: bool = MISSING
    remove_unused_columns: bool = MISSING
    bf16: bool = MISSING


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    checkpoint_name: Optional[str] = MISSING
    run_name: Optional[str] = MISSING
    seed: int = MISSING

    # derived
    model_load_path: str = "${mdlm_load_path:${.model.model_name},${.checkpoint_name}}"
    model_save_path: str = (
        "${mdlm_save_path:${.model.model_name},${.checkpoint_name},"
        "${.run_name},${.dataset.dataset_key}}"
    )
    tensorboard_log_dir: str = "${.model_save_path}/logs"


# ============================================================ #
# Registration
# ============================================================ #
def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="mdlm_train_config", node=TrainConfig)
    cs.store(group="model", name="mdlm-owt", node=ModelConfig(model_name="mdlm-owt"))
    for key in ("wrp", "alp", "tis"):
        cs.store(group="dataset", name=key, node=DatasetConfig(dataset_key=key))
