import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from flowdis.autoencoder import AutoEncoder
from flowdis.conditioner import HFEmbedder
from flowdis.configs import configs
from flowdis.loaders import load_autoencoder, load_clip, load_t5, load_transformer
from flowdis.model import Flux


logger = logging.getLogger(__name__)


@dataclass
class Models:
    clip: HFEmbedder
    t5: HFEmbedder
    ae: AutoEncoder
    transformer: Flux


def load_models(
    root_model_dir: Path = None,
    device: str | torch.device = "cuda"
) -> Models:
    """
    Load the models for the FlowDIS pipeline.
    
    Args:
        root_model_dir: The root model directory.
            If None, the models are downloaded from the Hugging Face Hub.
        device: The device to load the models on.
    
    Returns:
        Models: The loaded models.
    """
    if root_model_dir is None:
        root_model_dir = download_from_hf_hub("PAIR/FlowDIS")

    logger.info("Loading T5.")
    t5 = load_t5(
        model_path=root_model_dir / "t5-v1_1-xxl" / "model.safetensors",
        device=device,
        max_length=512
    )
    
    logger.info("Loading CLIP.")
    clip = load_clip(
        model_path=root_model_dir / "clip-vit-large-patch14" / "model.safetensors",
        device=device
    )
    
    logger.info("Loading AE.")
    ae = load_autoencoder(
        model_path=root_model_dir / "ae.safetensors", 
        device=device
    )
    
    logger.info("Loading Transformer.")
    model = load_transformer(
        model_name="flowdis",
        model_path=root_model_dir / "flowdis-transformer.safetensors",
        device=device,
    )

    logger.info("All models loaded.")
    
    return Models(
        clip=clip,
        t5=t5,
        ae=ae,
        transformer=model,
    )


def download_from_hf_hub(
    repo_id: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path:
    """
    Download a FlowDIS model repository from the Hugging Face Hub.

    Args:
        repo_id: The Hugging Face Hub repo id (e.g. "PAIR/FlowDIS").
        cache_dir: Optional cache directory. Defaults to the huggingface_hub
            default (typically ~/.cache/huggingface/hub).
        revision: Optional git revision (branch, tag, or commit SHA).

    Returns:
        Path to the local directory containing the downloaded snapshot. The
        directory layout matches the repo layout on the Hub, so it can be
        passed directly to `load_models` as `root_model_dir`.
    """
    logger.info(f"Downloading {repo_id} from Hugging Face Hub.")
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        revision=revision,
    )
    logger.info(f"Snapshot available at {local_dir}.")
    return Path(local_dir)


def green_screen(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_np = np.array(img)
    mask = (np.array(mask) / 255)[:, :, np.newaxis].repeat(3, axis=2)
    combined = img_np * mask + (1-mask) * np.array([0, 255, 0], dtype=np.uint8)
    combined = combined.astype(np.uint8)
    return combined
