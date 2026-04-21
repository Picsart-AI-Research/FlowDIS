
import torch
from safetensors.torch import load_file

from flowdis.autoencoder import AutoEncoder
from flowdis.conditioner import HFEmbedder
from flowdis.configs import configs
from flowdis.model import Flux, FluxParams


def load_transformer(
    model_name: str,
    model_path: str,
    device: str | torch.device = "cuda",
    config: FluxParams = None,
    state_dict: dict = None,
) -> Flux:
    with torch.device("meta"):
        model = Flux(config if config else configs[model_name]).to(dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    if state_dict is None:
        if str(model_path).endswith(".safetensors"):
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, assign=True, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    return model.eval()


def load_autoencoder(
    model_path: str,
    device: str | torch.device = "cuda"
) -> AutoEncoder:
    with torch.device("meta"):
        ae = AutoEncoder(configs["autoencoder"])
    ae.to_empty(device="cpu")
    state_dict = load_file(model_path, device="cpu")
    ae.load_state_dict(state_dict, assign=True, strict=False)
    ae = ae.to(device=device, dtype=torch.bfloat16)
    return ae.eval()


def load_t5(
    model_path: str,
    max_length: int = 512,
    device: str | torch.device = "cuda"
) -> HFEmbedder:
    with torch.device("meta"):
        t5 = HFEmbedder(
            model_path.parent,
            max_length=max_length,
            is_clip=False,
            dtype=torch.bfloat16
        )
    t5.to_empty(device="cpu")
    state_dict = load_file(model_path, device="cpu")
    t5.load_state_dict(state_dict, assign=True, strict=False)
    return t5.to(device=device, dtype=torch.bfloat16)


def load_clip(
    model_path: str,
    device: str | torch.device = "cuda"
) -> HFEmbedder:
    clip = HFEmbedder(
        model_path.parent,
        max_length=77,
        is_clip=True,
        dtype=torch.bfloat16
    )
    state_dict = load_file(model_path, device="cpu")
    clip.load_state_dict(state_dict, assign=True, strict=False)
    return clip.to(device=device, dtype=torch.bfloat16)
    
