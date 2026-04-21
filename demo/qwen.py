import logging

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

logger = logging.getLogger(__name__)

# Load model
logger.info("Loading Qwen3VL model.")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
logger.info("Qwen3VL model loaded.")

def expand_prompt(image: Image.Image, user_prompt: str) -> str:
    """
    Expand the user prompt using the Qwen3VL model.
    
    Args:
        image: The image to use for the prompt expansion.
        user_prompt: The user prompt to expand.
    
    Returns:
        The expanded prompt.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Describe the {user_prompt} in this image with as a short prompt. Don't use surrounding objects in the description. Also don't describe the background, like what it is sitting on or what it is on top of, etc..."}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512
        )
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    return output_text
