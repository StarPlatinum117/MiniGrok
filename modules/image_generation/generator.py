import logging
import pathlib
from diffusers import StableDiffusionPipeline

from modules.config import IMAGE_GENERATION_MODEL_NAME as MODEL_NAME
from modules.config import IMAGE_GENERATION_IMAGES_DIR as IMAGES_DIR
from modules.image_generation.model_loader import load_model
import torch
from PIL import Image
from PIL import ImageDraw
import random


def generate_image(
        *,
        model_name: str,
        prompt: str,
        output_dir: pathlib.Path,
        device: str = "cpu"
) -> Image.Image:
    """
    Generate an image from a text prompt using either the Stable Diffusion model or the dummy model.

    Args:
        model_name: The name of the Stable Diffusion model to use.
        prompt: The text prompt to generate the image from.
        output_dir: The dir to save the generated image.
        device: The device to run the model on (default is "cpu").

    Returns:
        The generated image.
    """
    if model_name == "dummy":
        logging.info(
            "Dummy model activated. The generated image will be unrelated to the the following prompt:\n"
            f"{prompt}"
        )
        size = (512, 512)
        image = Image.new("RGB", size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Placeholder image", fill=(255, 255, 255))
        return image

    else:
        logging.info(f"Generating image with prompt: {prompt}")

        # Load the model.
        pipe = load_model(model_name=model_name, device=device)
        logging.info(f"Model {model_name} loaded successfully on {device}.")

        # Generate the image.
        image = pipe(prompt).images[0]

    logging.info("Image generated successfully.")

    # Save the image to the output path.
    file_path = output_dir / f"output_image_{model_name}.png"
    image.save(file_path)
    logging.info(f"Image saved to {file_path}")

    return image


if __name__ == "__main__":
    # Example usage. Currently, only "dummy" model works since diffusion models cannot be downloaded.
    generated_image = generate_image(
        model_name=MODEL_NAME,
        prompt="A beautiful landscape with mountains and a river",
        output_dir=IMAGES_DIR,
        device="cpu"
    )

