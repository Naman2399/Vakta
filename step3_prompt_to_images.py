
import torch

# Monkeypatch: Some libs use torch._C.TensorBase which is removed in new torch
if not hasattr(torch._C, "TensorBase"):
    torch._C.TensorBase = torch.Tensor


from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from PIL import Image
import requests
import os

def load_image(path_or_url):
    """Helper to load local or remote image."""
    if path_or_url.startswith("http"):
        return Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
    else:
        return Image.open(path_or_url).convert("RGB")

def generate_scene(prompt, char1_path, char2_path, output_path="output.png"):
    """
    Generate scene with two consistent characters using IP-Adapter + SDXL.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Stable Diffusion XL ---
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)

    # Optional: better quality with custom VAE
    pipe.vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float16
    ).to(device)

    # --- Load character face images ---
    char1_img = load_image(char1_path)
    char2_img = load_image(char2_path)

    # --- Merge both references side by side ---
    w, h = char1_img.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(char1_img, (0, 0))
    combined.paste(char2_img, (w, 0))

    # --- Run pipeline with reference ---
    images = pipe(
        prompt=prompt,
        image=combined,   # reference images
        strength=0.7,     # how much to follow reference faces
        guidance_scale=7.5,
        num_inference_steps=30
    ).images

    # --- Save result ---
    images[0].save(output_path)
    print(f"✅ Scene saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    char1 = "books/modi.jpeg"   # Replace with your character 1 face image
    char2 = "books/putin.jpeg"   # Replace with your character 2 face image

    scene_prompt = "A cinematic illustration of two astronauts, character1 and character2, exploring space with stars in the background, futuristic, highly detailed"

    generate_scene(scene_prompt, char1, char2, output_path="space_scene.png")
