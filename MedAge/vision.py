import logging
import os
from typing import List, Tuple
from PIL import Image
import torch

from MedAge.config import _model_cache, DEVICE

logger = logging.getLogger(__name__)

def _load_vision_model(model_name: str = "Haider584/lora_model"):
    """
    Load and cache the vision model to avoid reloading on every call.
    """
    if (_model_cache["model"] is None or
        _model_cache["model_name"] != model_name):
        logger.info(f"🔄 Loading vision model: {model_name}")
        from unsloth import FastVisionModel  # local import to avoid heavy import at module load
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(model)
        try:
            model.to(DEVICE)
        except Exception:
            pass
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
        _model_cache["model_name"] = model_name
        logger.info(f"✅ Model loaded and cached on {DEVICE}")
    else:
        logger.info(f"✅ Using cached model on {DEVICE}")
    return _model_cache["model"], _model_cache["tokenizer"]

def _load_single_image(image_path: str) -> Image.Image:
    normalized_path = image_path.strip().strip('"').strip("'")
    if not os.path.exists(normalized_path):
        if not os.path.isabs(normalized_path):
            cwd_path = os.path.join(os.getcwd(), normalized_path)
            if os.path.exists(cwd_path):
                normalized_path = cwd_path
            else:
                raise FileNotFoundError(f"Image not found at: {normalized_path} or {cwd_path}")
        else:
            raise FileNotFoundError(f"Image not found at: {normalized_path}")
    try:
        pil_image = Image.open(normalized_path).convert("RGB")
        logger.info(f" ✅ Loaded image: {normalized_path} (size: {pil_image.size})")
        return pil_image
    except Exception as e:
        raise ValueError(f"Failed to open image {normalized_path}: {e}")

def _load_multiple_images(image_paths: List[str]):
    pil_images = []
    logger.info(f"🔄 Loading {len(image_paths)} image(s)...")
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f" Loading image {i}/{len(image_paths)}...")
        pil_image = _load_single_image(image_path)
        pil_images.append(pil_image)
    logger.info(f"✅ Successfully loaded {len(pil_images)} image(s)")
    return pil_images

def _prepare_model_inputs_multiple(question: str, pil_images: List[Image.Image], tokenizer):
    content = [{"type": "text", "text": question}]
    for _ in pil_images:
        content.append({"type": "image", "image": "<image>"})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    images_for_batch = [pil_images]
    inputs = tokenizer(
        text=[input_text],
        images=images_for_batch,
        return_tensors="pt",
        padding=True,
    )
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(DEVICE)
    return inputs, input_text

def _generate_answer(model, inputs, max_new_tokens=512, temperature=0.0001):
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=True,
        )
    return output

def _decode_answer(output, tokenizer, input_text: str = None) -> str:
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "<|im_end|>" in decoded:
        decoded = decoded.split("<|im_end|>")[0]
    if input_text and input_text in decoded:
        decoded = decoded.split(input_text, 1)[-1]
    return decoded.strip()
