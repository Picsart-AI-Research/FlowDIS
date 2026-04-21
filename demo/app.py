import csv
import os
import logging
import uuid
import shutil
from copy import deepcopy
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set Gradio temp directory BEFORE importing gradio to avoid permission issues
TEMP_DIR = Path(__file__).parent / "gradio_temp"
if TEMP_DIR.exists():
    shutil.rmtree(str(TEMP_DIR))
TEMP_DIR.mkdir(exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)
os.environ["TMPDIR"] = str(TEMP_DIR)

import gradio as gr
import numpy as np
import torch
from PIL import Image

from flowdis.sampling import flowdis_predict
from flowdis.util import load_models
from qwen import expand_prompt


device = "cuda"
models = load_models(device=device)


def disable_download_btn():
    return gr.update(interactive=False)


def process_image(image, prompt, expand_prompt_enabled, resolution, num_inference_steps):
    """
    Process the input image and prompt.
    This is a placeholder function - replace with your actual processing logic.
    
    Args:
        image: PIL Image or numpy array
        prompt: str, the text input from the user
        expand_prompt_enabled: bool, whether to expand the prompt via the model
        resolution: int, the inference resolution
        num_inference_steps: int, the number of inference steps
    
    Returns:
        Processed image
    """
    if image is None:
        return None, None
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    logger.info(f"Original prompt: {prompt}")
    if prompt != "" and expand_prompt_enabled:
        prompt = expand_prompt(image, prompt)
        logger.info(f"Expanded prompt: {prompt}")

    num_inference_steps = int(num_inference_steps)

    pred_mask = flowdis_predict(
        image=image,
        prompt=prompt,
        models=models,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    blacked_image = Image.fromarray(np.array(image) * (np.array(pred_mask)[:, :, np.newaxis] > 0).astype(np.uint8))
    transparent_png = Image.fromarray(np.dstack([blacked_image, np.array(pred_mask)]))
    uid = uuid.uuid4().hex
    png_path = TEMP_DIR / f"{uid}.png"
    transparent_png.save(png_path)
    return (
        gr.update(value=[image, transparent_png], key=uid),
        gr.update(value=str(png_path), interactive=True)
    )


# Load examples from assets/examples/examples.csv: image_name, prompt, resolution, num_steps
_example_dir = Path(__file__).parent.parent / "assets" / "examples"
_examples_csv = _example_dir / "examples.csv"
examples = []
if _examples_csv.exists():
    with open(_examples_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = str(_example_dir / row["image_name"].strip())
            examples.append([
                image_path,
                row["prompt"].strip(),
                True,  # expand prompt (default for examples)
                int(row["resolution"].strip()),
                int(row["num_steps"].strip()),
            ])


_head_js = """
<style>
#expand-prompt.is-disabled { pointer-events: none !important; }
#expand-prompt.is-disabled label,
#expand-prompt.is-disabled input,
#expand-prompt.is-disabled .info { opacity: 0.4 !important; }
/* Hide the "Expand prompt" column (3rd) in the examples table */
#examples-table table th:nth-child(3),
#examples-table table td:nth-child(3) { display: none !important; }
</style>
<script>
(function() {
    function findEls() {
        return {
            ta: document.querySelector('#text-prompt textarea, #text-prompt input'),
            cb: document.querySelector('#expand-prompt'),
        };
    }
    function syncFromText() {
        var els = findEls();
        if (!els.ta || !els.cb) return;
        var empty = !els.ta.value.trim();
        els.cb.classList.toggle('is-disabled', empty);
        var input = els.cb.querySelector('input[type=checkbox]');
        if (input) input.disabled = empty;
    }
    function init() {
        var els = findEls();
        if (!els.ta || !els.cb) { setTimeout(init, 200); return; }
        els.ta.addEventListener('input', syncFromText);
        els.ta.addEventListener('change', syncFromText);
        // Catch programmatic value changes (e.g. example selection)
        var lastVal = els.ta.value;
        setInterval(function() {
            if (els.ta.value !== lastVal) { lastVal = els.ta.value; syncFromText(); }
        }, 250);
        syncFromText();
    }
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', init);
    else
        init();
})();
</script>
<script>
(function() {
    function findEls() {
        return {
            ta: document.querySelector('#text-prompt textarea, #text-prompt input'),
            cb: document.querySelector('#expand-prompt'),
        };
    }
    function syncFromText() {
        var els = findEls();
        if (!els.ta || !els.cb) return;
        var empty = !els.ta.value.trim();
        els.cb.classList.toggle('is-disabled', empty);
        var input = els.cb.querySelector('input[type=checkbox]');
        if (input) input.disabled = empty;
    }
    function init() {
        var els = findEls();
        if (!els.ta || !els.cb) { setTimeout(init, 200); return; }
        els.ta.addEventListener('input', syncFromText);
        els.ta.addEventListener('change', syncFromText);
        // Catch programmatic value changes (e.g. example selection)
        var lastVal = els.ta.value;
        setInterval(function() {
            if (els.ta.value !== lastVal) { lastVal = els.ta.value; syncFromText(); }
        }, 250);
        syncFromText();
    }
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', init);
    else
        init();
})();
</script>
"""
with gr.Blocks(
    title="FlowDIS – Precise Background Removal",
    head=_head_js,
    theme=gr.themes.Default(
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        button_primary_background_fill="#C209C1",
        button_primary_background_fill_dark="#C209C1",
        button_primary_background_fill_hover="#d63bd5",
        button_primary_background_fill_hover_dark="#d63bd5",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
    ),
    delete_cache=(1800, 1800)
) as demo:
    gr.HTML(
        """
       <style>
        /* Theme-adaptive tokens */
        :root {
        --flow-text: #0f172a;          /* slate-900 */
        --flow-muted: #475569;         /* slate-600 */
        --flow-link: #2563eb;          /* blue-600 */
        --flow-link-hover: #1d4ed8;    /* blue-700 */
        --flow-title: #C209C1;         /* Picsart pink */
        }

        @media (prefers-color-scheme: dark) {
        :root {
            --flow-text: #f1f5f9;        /* slate-100 */
            --flow-muted: #94a3b8;       /* slate-400 */
            --flow-link: #60a5fa;        /* blue-400 */
            --flow-link-hover: #93c5fd;  /* blue-300 */
            --flow-title: #e45fe3;       /* Picsart pink (lighter for dark mode) */
        }
        }

        .flow-header {
        text-align: center;
        max-width: 900px;
        margin: 18px auto 12px auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .flow-title {
        font-size: 1.9rem;
        font-weight: 750;
        letter-spacing: -0.3px;
        margin-bottom: 4px;
        color: var(--flow-title);      /* title accent (needle stays as-is) */
        }

        .flow-links {
        margin-bottom: 8px;
        }

        .flow-links a {
        color: var(--flow-link);       /* cool blue links */
        font-weight: 600;
        text-decoration: none;
        margin: 0 0px;
        font-size: 0.95rem;
        transition: color 0.2s ease, text-shadow 0.2s ease;
        }

        .flow-links a:hover {
        color: var(--flow-link-hover);
        text-shadow: 0 0 10px rgba(37, 99, 235, 0.25);
        }

        @media (prefers-color-scheme: dark) {
        .flow-links a:hover {
            text-shadow: 0 0 12px rgba(147, 197, 253, 0.35);
        }
        }

        .flow-desc {
        font-size: 0.95rem;
        color: var(--flow-muted);
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.5;
        }

        .bg-btn-row { display: flex; gap: 6px; overflow-x: auto; scrollbar-width: thin; }
        .bg-btn {
            width: 42px !important; height: 42px !important;
            border: 2.5px solid #aaa !important; border-radius: 8px !important;
            cursor: pointer !important; flex-shrink: 0 !important;
            padding: 0 !important; outline: none !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease,
                        border-color 0.15s ease, filter 0.15s ease;
        }
        .bg-btn:hover {
            transform: scale(1.15);
            border-color: #333 !important;
            box-shadow: 0 3px 10px rgba(0,0,0,0.4);
            filter: brightness(1.15);
        }
        .bg-btn:active {
            transform: scale(0.95);
        }


        @media (max-width: 1024px) {
            #main-row {
                flex-direction: column !important;
                flex-wrap: wrap !important;
            }
            #main-row > * {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 0 !important;
            }
        }

        @media (max-width: 500px) {
            #input-image { height: 400px !important; }
        }
        @media (max-width: 400px) {
            #input-image { height: 300px !important; }
        }
        .prose :is(label span, .info) { font-weight: 400 !important; }
        </style>

        <div class="flow-header">
        <div class="flow-title"><span style="color:#C209C1">✦</span> FlowDIS Demo</div>

        <div class="flow-links">
            <span>📄</span><a href="https://arxiv.org/" target="_blank" rel="noopener noreferrer">arXiv</a>
            <span>💻</span><a href="https://github.com/Picsart-AI-Research/FlowDIS" target="_blank" rel="noopener noreferrer">Code</a>
        </div>

        <div class="flow-desc">
            FlowDIS performs precise foreground segmentation, optionally guided by a text prompt to only preserve the specified objects.
        </div>
        </div>
        """
    )
    
    with gr.Row(elem_id="main-row"):
        # Left column: Input image, text field, and submit button
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="pil",
                height=500,
                elem_id="input-image",
            )
            text_input = gr.Textbox(
                label="Text Prompt (Optional)",
                placeholder="Enter what you want to retain...",
                lines=1,
                elem_id="text-prompt",
            )
            expand_prompt_check = gr.Checkbox(
                label="Expand prompt",
                value=True,
                elem_id="expand-prompt",
                info="Use Qwen3-VL-4B-Instruct model to expand the prompt for better text-guided segmentation.",
            )
            
            # Sliders for resolution and steps
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    resolution_slider = gr.Slider(
                        minimum=1024,
                        maximum=2048,
                        value=1536,
                        step=64,
                        label="Inference Resolution",
                        info="Higher resolution preserves more details.",
                    )
                    
                with gr.Column(scale=1, min_width=300):
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=4,
                        step=1,
                        label="Number of Steps",
                        info="More steps generate sharper results.",
                    )
            
            submit_btn = gr.Button("🚀 Remove Background", variant="primary")
        
        # Right column: Output image
        with gr.Column(scale=1):
            output_image = gr.ImageSlider(
                label="FlowDIS prediction",
                type="pil",
                format="webp",
                height=500,
                slider_position=10,
                elem_id="output-slider",
            )

            _checker = "repeating-conic-gradient(#ccc 0% 25%,#fff 0% 50%) 50%/12px 12px"
            _bg_buttons = [
                (_checker, _checker),
                ("#ffffff", "#ffffff"),
                ("#000000", "#000000"),
                ("#00ff00", "#00ff00"),
                ("#0000ff", "#0000ff"),
                ("#ff0000", "#ff0000"),
                ("#ffff00", "#ffff00"),
                ("#ff00ff", "#ff00ff"),
                ("#00ffff", "#00ffff"), 
            ]
            _onclick = (
                "var s=document.getElementById('slider-bg-style');"
                "if(!s){s=document.createElement('style');"
                "s.id='slider-bg-style';document.head.appendChild(s);}"
                "s.textContent='#output-slider img,#output-slider canvas"
                "{background:'+this.dataset.bg+' !important}';"
            )
            gr.HTML(
                value='<div class="bg-btn-row">'
                + "".join(
                    f'<button class="bg-btn" style="background:{style}"'
                    f' data-bg="{bg}" onclick="{_onclick}"></button>'
                    for style, bg in _bg_buttons
                )
                + "</div>"
            )

            download_btn = gr.DownloadButton(
                label="📥 Download PNG",
                variant="primary",
                interactive=False
            )
    
    # Connect the submit button to the processing function
    submit_btn.click(
        disable_download_btn,
        outputs=download_btn
    ).then(
        fn=process_image,
        inputs=[input_image, text_input, expand_prompt_check, resolution_slider, steps_slider],
        outputs=[output_image, download_btn]
    )
    
    # Optional: Also trigger on text input enter key
    text_input.submit(
        disable_download_btn,
        outputs=download_btn
    ).then(
        fn=process_image,
        inputs=[input_image, text_input, expand_prompt_check, resolution_slider, steps_slider],
        outputs=[output_image, download_btn],
    )

    examples_component = gr.Examples(
        examples=examples,
        inputs=[input_image, text_input, expand_prompt_check, resolution_slider, steps_slider],
        label="Examples",
        elem_id="examples-table",
    )

    examples_component.dataset.click(
        disable_download_btn,
        outputs=download_btn
    ).then(
        process_image,
        inputs=[input_image, text_input, expand_prompt_check, resolution_slider, steps_slider],
        outputs=[output_image, download_btn],
    )


# Launch the app
if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(
        server_name="0.0.0.0",  # Makes it accessible on your network
        server_port=7860,
        share=True,  # Set to True if you want a public share link
        allowed_paths=[str(TEMP_DIR), "assets"]
    )

