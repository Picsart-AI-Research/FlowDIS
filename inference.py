import argparse
import json
import logging
import os
from dataclasses import fields
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from flowdis.sampling import flowdis_predict
from flowdis.util import Models, load_models


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="FlowDIS inference")
    parser.add_argument(
        "--root-model-dir",
        type=Path,
        required=False,
        default=None,
        help="Root model directory. If omitted, the weights are downloaded "
             "from the Hugging Face Hub (PAIR/FlowDIS) and cached locally."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Path to the images directory to use for inference"
    )
    parser.add_argument(
        "--prompts-json",
        type=Path,
        required=False,
        default=None,
        help="Path to the prompts json file"
    )
    parser.add_argument(
        "--num-steps",
        default=2,
        type=int,
        required=False,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolution of the image"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of images to process"
    )
    return parser.parse_args()


def process(
    gpu_id: int,
    img_paths: list[Path],
    models: Models,
    prompts: dict[str, str],
    args: argparse.Namespace
) -> None:
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    models = Models(**{field.name: getattr(models, field.name).to(device=device) for field in fields(models)})
    
    for img_path in tqdm(img_paths):
        image = Image.open(img_path).convert("RGB")
        prompt = prompts[Path(img_path).name]
        pred_mask = flowdis_predict(
            image=image,
            prompt=prompt,
            models=models,
            resolution=args.resolution,
            num_inference_steps=args.num_steps,
            device=device,
        )
        pred_mask.save(args.output_dir / f"{Path(img_path).stem}.png")


def main():
    args = get_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    img_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(args.images_dir, "**", f"*{img_extension}"), recursive=True)
    
    img_paths = sorted(img_paths)
    
    if args.prompts_json is not None:
        with open(args.prompts_json, "r") as f:
            prompts = json.load(f)
    else:
        prompts = {Path(img_path).name: "" for img_path in img_paths}
    
    logger.info(f"Processing {len(img_paths)} images")

    models = load_models(root_model_dir=args.root_model_dir, device="cpu")
    
    if args.num_samples > 0:
        img_paths = img_paths[:args.num_samples]
       
    num_gpus = torch.cuda.device_count()
    chunks = [img_paths[i::num_gpus] for i in range(num_gpus)]
    
    logger.info(f"Inference will start shortly on {num_gpus} GPUs...")


    if num_gpus == 1:
        process(0, chunks[0], models, prompts, args)
    else:
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=process, args=(i, chunks[i], models, prompts, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    
if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
