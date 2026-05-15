# FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching

<div align="center">
  <a href="https://scholar.google.com/citations?user=cg74A98AAAAJ&hl=en">Andranik Sargsyan</a>, <a href="https://scholar.google.com/citations?user=VJSh59sAAAAJ&hl=en">Shant Navasardyan</a>
</div>
<div align="center">
  Picsart AI Research (PAIR)
</div>

<br/>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap; margin-bottom: 3px;"> 
  <a href='https://arxiv.org/abs/2605.05077'><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arXiv&logoColor=red'></a>&ensp; 
  <a href='https://flowdis.github.io/'><img src='https://img.shields.io/badge/Page-Project-blue'></a>&ensp; 
  <a href="https://huggingface.co/spaces/PAIR/FlowDIS"><img src="https://img.shields.io/badge/Hugging%20Face-Space-yellow?logo=huggingface" alt="HuggingFace Space"></a>&ensp;
  <a href='https://drive.google.com/drive/folders/1xWRlwnNbfELDqduPxJ6Z04xP2-9zfaJj?usp=sharing'><img src='https://img.shields.io/badge/Google%20Drive-Stuff-green?logo=googledrive&logoColor=green'></a>&ensp;
</div>

<br/>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="assets/teaser-dark.avif">
    <source media="(prefers-color-scheme: light)" srcset="assets/teaser-light.avif">
    <img src="assets/teaser-light.avif" alt="FlowDIS teaser" width="100%">
  </picture>
</p>

<p align="center">
<i>FlowDIS enables highly accurate foreground segmentation, optionally guided by a text prompt. When ambiguity prevents the model from producing the desired result, the user can specify which elements to retain in the foreground.</i>
<p>

## News
- **[May 15, 2026]** 🤗 Hugging Face Space released.
- **[May 6, 2026]** 📄 Paper released on arXiv.
- **[February 21, 2026]** 🎉 FlowDIS accepted to CVPR 2026.

## Requirements

- Python **3.12** (the project is tested on 3.12)
- A CUDA-capable GPU (multi-GPU inference is supported)

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/Picsart-AI-Research/FlowDIS
cd FlowDIS
pip install -e .
```

## Models

Model weights are hosted on the Hugging Face Hub at [`PAIR/FlowDIS`](https://huggingface.co/PAIR/FlowDIS). They are downloaded automatically on first run and cached under `~/.cache/huggingface/hub`.

To pre-download manually:

```python
from flowdis.util import download_from_hf_hub
root_model_dir = download_from_hf_hub("PAIR/FlowDIS")
print(root_model_dir)
```

## Inference

Run inference on a directory of images. If multiple GPUs are available, the workload is automatically split across them.

```bash
python inference.py \
    --images-dir /path/to/images \
    --output-dir /path/to/output \
    --prompts-json /path/to/prompts.json \
    --num-steps 2 \
    --resolution 1024
```

To use local weights instead of auto-downloading, pass `--root-model-dir`:

```bash
python inference.py \
    --root-model-dir /path/to/models \
    --images-dir /path/to/images \
    --output-dir /path/to/output
```

### Arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--images-dir` | yes | – | Directory of input images (`.jpg`, `.jpeg`, `.png`); searched recursively. |
| `--output-dir` | yes | – | Directory where predicted masks (`.png`) are written. |
| `--root-model-dir` | no | `None` | Root directory of pre-downloaded weights. If omitted, weights are fetched from `PAIR/FlowDIS` on the Hugging Face Hub. |
| `--prompts-json` | no | `None` | JSON mapping `{ "image_filename": "prompt" }`. If omitted, empty prompts are used. |
| `--num-steps` | no | `2` | Number of flow-matching sampling steps. |
| `--resolution` | no | `1024` | Image resolution used for inference. |

### Prompts file format

```json
{
    "image_001.jpg": "a red sports car",
    "image_002.png": "a golden retriever sitting on grass"
}
```

Pre-generated language prompts for the DIS dataset are available [here](https://drive.google.com/drive/folders/1ikCxZeJZcwcSHs1_EOQPRRM_Rb9q1pXo?usp=sharing). Precomputed results for reproducing the paper can be downloaded [here](https://drive.google.com/drive/folders/1PPMabkVyT2IQ-oE_t1mA92U9q-dfCZcC?usp=sharing).
 

## Demo

An interactive Gradio demo is included under `demo/`:

```bash
python demo/app.py
```

Hardware requirements:

- **At least 48 GB of GPU memory** for inference at 1024×1024px.
- **80 GB of GPU memory** is required for inference at higher resolutions (such as 2048x2048px).

## Programmatic usage

```python
from PIL import Image
from flowdis import flowdis_predict, load_models

models = load_models(device="cuda")

input_img_path = "path/to/input.jpg"     # Input image path
output_mask_path = "path/to/output.png"  # Path to save the output mask

image = Image.open(input_img_path).convert("RGB")

mask = flowdis_predict(
    image=image,
    prompt="",  # Text prompt
    models=models,
    resolution=1024,
    num_inference_steps=2,
    device="cuda",
)
mask.save(output_mask_path)
```

## License

FlowDIS is licensed under the [PicsArt Inc. FlowDIS Model License](https://github.com/Picsart-AI-Research/FlowDIS/blob/main/LICENSE).

## Acknowledgements

This project is built on top of [FLUX.1 [schnell]](https://github.com/black-forest-labs/flux) and [DIS5K](https://github.com/xuebinqin/DIS). 


## BibTeX

If you use our work in your research, please cite our publication:

```
@article{sargsyan2026flowdis,
  title={{FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching}},
  author={Sargsyan, Andranik and Navasardyan, Shant},
  journal={arXiv preprint arXiv:2605.05077},
  year={2026},
  eprint={2605.05077},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.05077}
}
```
