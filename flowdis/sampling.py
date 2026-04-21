import math

import torch
import torchvision.transforms.functional as tvF
from einops import rearrange, repeat
from PIL import Image
from scipy import stats
from torch import Tensor

from flowdis.model import Flux
from flowdis.util import Models


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def beta_scheduler(num_timesteps: int, alpha: float = 2.5, beta: float = 1.0) -> list[float]:
    q = torch.linspace(1, 0, num_timesteps+1)
    steps = stats.beta.ppf(q, alpha, beta).tolist()
    if steps[-1] > 0.0:
        steps.append(0.0)
    return steps


def prepare(
    img: Tensor,
    prompt: str | list[str],
    models: Models,
    device: str = "cuda"
) -> dict[str, Tensor]:
    # load and encode the conditioning image and the mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)
    if isinstance(prompt, str):
        prompt = [prompt]

    with torch.no_grad():
        img = models.ae.encode(img.to(device=device, dtype=torch.bfloat16))
    h, w = img.shape[2], img.shape[3]
    
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)    
    
    txt = models.t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)
    
    vec = models.clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return_dict = {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

    return return_dict


def solve_flowdis_ode(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    num_inference_steps: int,
):
    zt = img
    timesteps = beta_scheduler(num_inference_steps)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):        
        t_vec = torch.full((zt.shape[0],), t_curr, dtype=zt.dtype, device=zt.device)
        pred = model(
            img=torch.cat((zt, img), dim=-1),
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
        )
        zt = zt + (t_prev - t_curr) * pred
    return zt


@torch.no_grad()
def flowdis_predict(
    image: Tensor,
    prompt: str | list[str],
    models: Models,
    resolution: int = 1024,
    num_inference_steps: int = 2,
    device: str = "cuda",
):
    image_orig = image.convert("RGB")
    image = image.resize((resolution, resolution))

    image_t = tvF.to_tensor(image).unsqueeze(0).to(device=device)
    image_t = (image_t - 0.5) / 0.5
    
    inp = prepare(image_t, prompt, models, device)

    pred_mask_latent_t = solve_flowdis_ode(
        models.transformer, 
        **inp, 
        num_inference_steps=num_inference_steps,
    )

    pred_mask_latent_t = unpack(pred_mask_latent_t.float(), resolution, resolution)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        pred_mask_t = models.ae.decode(pred_mask_latent_t).clamp(-1, 1)
    
    pred_mask_t = rearrange(pred_mask_t[0], "c h w -> h w c")
    pred_mask_np = (127.5 * (pred_mask_t + 1.0)).mean(dim=-1).cpu().byte().numpy()
    pred_mask = Image.fromarray(pred_mask_np).convert("L")
    pred_mask = pred_mask.resize(image_orig.size)

    return pred_mask
