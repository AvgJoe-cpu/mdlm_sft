from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Resolve nested artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"

# Config directories
CONFIG_DIR = PROJECT_ROOT / "config"
AR_CONFIG_DIR = CONFIG_DIR / "ar"
MDLM_CONFIG_DIR = CONFIG_DIR / "mdlm"

# Dataset directory structure
DATASET_BASE_DIR = DATASETS_DIR / "base"  # Shared base datasets
DATASET_GEN_DIR = DATASETS_DIR / "gen"     # Model-specific generated datasets

# Dataset metadata (model-agnostic)
DATASETS = {
    "wrp": {
        "name": "Writing-prompts",
        "link": "https://huggingface.co/datasets/euclaise/writingprompts",
        "hf-path": "euclaise/writingprompts",
        "base_path": DATASET_BASE_DIR / "wrp",  # Shared across all models
    },    
    "alp": {
        "name": "Alpaca",
        "link": "https://huggingface.co/datasets/tatsu-lab/alpaca",
        "hf-path": "tatsu-lab/alpaca",
        "base_path": DATASET_BASE_DIR / "alp",
    },
    "tis": {
        "name": "TinyStories",
        "link": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "hf-path": "roneneldan/TinyStories",
        "base_path": DATASET_BASE_DIR / "tis",
    },
    "tms": {
        "name": "Tell me a Story",
        "link": "https://github.com/google-deepmind/tell_me_a_story",
        "hf-path": None,  # Not on Hugging Face - will load from local
        "base_path": DATASET_BASE_DIR / "tms",
    },
}


def resolve_dataset_base_path(dataset_key: str) -> Path:
    """Get the base (raw) dataset path - shared by all models"""
    dataset_info = DATASETS.get(dataset_key)
    if not dataset_info:
        raise ValueError(f"Dataset key '{dataset_key}' not found in DATASETS.")
    
    base_path = dataset_info["base_path"]
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base path for dataset '{dataset_key}' does not exist: {base_path}"
        )
    
    return base_path


def get_gen_dataset_path(dataset_key: str, model_type: str) -> Path:
    """
    Get the generated dataset path for a specific model type.
    
    Structure: artifacts/datasets/gen/{model_type}/{dataset_key}/
    
    Args:
        dataset_key: Dataset key (e.g., "wrp", "alp", "tis")
        model_type: Model type (e.g., "ar", "mdlm")
    
    Returns:
        Path to model-specific generated dataset directory
    """
    if dataset_key not in DATASETS:
        raise ValueError(
            f"Dataset key '{dataset_key}' not found in DATASETS. "
            f"Available: {list(DATASETS.keys())}"
        )
    
    return DATASET_GEN_DIR / model_type / dataset_key


### MODELS 
WEIGHTS_DIR = ARTIFACTS_DIR / "weights"

# AR Models
AR_MODELS_DIR = WEIGHTS_DIR / "ar"
AR_MODEL_BASE_DIR = AR_MODELS_DIR / "base"
AR_MODEL_CHECKPOINTS_DIR = AR_MODELS_DIR / "checkpoints"

AR_MODELS = {
    "pythia-70m": {
        "name": "Pythia-70M",
        "link": "https://huggingface.co/EleutherAI/pythia-70m",
        "hf-path": "EleutherAI/pythia-70m",
        "base_path": AR_MODEL_BASE_DIR / "pythia-70m",
        "checkpoints_path": AR_MODEL_CHECKPOINTS_DIR / "pythia-70m",
        "model_type": "ar",  # Used for dataset gen path resolution
    },
    "pythia-160m": {
        "name": "Pythia-160M",
        "link": "https://huggingface.co/EleutherAI/pythia-160m",
        "hf-path": "EleutherAI/pythia-160m",
        "base_path": AR_MODEL_BASE_DIR / "pythia-160m",
        "checkpoints_path": AR_MODEL_CHECKPOINTS_DIR / "pythia-160m",
        "model_type": "ar",
    },
}

# MDLM Models
MDLM_MODELS_DIR = WEIGHTS_DIR / "mdlm"
MDLM_MODEL_BASE_DIR = MDLM_MODELS_DIR / "base"
MDLM_MODEL_CHECKPOINTS_DIR = MDLM_MODELS_DIR / "checkpoints"

MDLM_MODELS = {
    "mdlm-owt": {
        "name": "MDLM-OWT",
        "link": "https://huggingface.co/avgJo3/mdlm-owt-bucket",
        "hf-path": "avgJo3/mdlm-owt-bucket",
        "base_path": MDLM_MODEL_BASE_DIR / "mdlm-owt",
        "checkpoints_path": MDLM_MODEL_CHECKPOINTS_DIR / "mdlm-owt",
        "model_type": "mdlm",
    },
}