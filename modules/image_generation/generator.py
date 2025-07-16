import logging
import pathlib
import random
from hashlib import sha1

from diffusers import StableDiffusionPipeline
from PIL import Image
from PIL import ImageDraw

from modules.config import IMAGE_GENERATION_IMAGES_DIR as IMAGES_DIR
from modules.config import IMAGE_GENERATION_MODEL_NAME as MODEL_NAME
from modules.image_generation.model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_image(
        *,
        model: StableDiffusionPipeline | str,
        prompt: str,
        output_dir: pathlib.Path,
        device: str = "cpu"
) -> dict[str, pathlib.Path]:
    """
    Generate an image from a text prompt using either the Stable Diffusion model or the dummy model.

    Args:
        model: The loaded StableDiffussionPipeline model or name of the model.
        prompt: The text prompt to generate the image from.
        output_dir: The dir to save the generated image.
        device: The device to run the model on (default is "cpu").

    Returns:
        A dictionary with the path to the generated image.
    """
    if model == "dummy":
        logging.info(
            "Dummy model activated. The generated image will be unrelated to the the following prompt:\n"
            f"{prompt}"
        )
        size = (512, 512)
        image = Image.new("RGB", size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Placeholder image", fill=(255, 255, 255))

    else:
        logging.info(f"Generating image with prompt: {prompt}")

        # Generate the image.
        image = model(prompt).images[0]

    logging.info("Image generated successfully.")

    # Save the image to the output path.
    hashed = sha1(prompt.encode()).hexdigest()[:8]
    model_tag = "dummy" if model == "dummy" else "real"
    file_path = output_dir / f"img_{model_tag}_{hashed}.png"
    file_path.parent.mkdir(exist_ok=True, parents=True)
    image.save(file_path)
    logging.info(f"Image saved to {file_path}")

    return {"image_path": file_path}


if __name__ == "__main__":
    # Example usage. Currently, only "dummy" model works since diffusion models cannot be downloaded.
    generated_image = generate_image(
        model=MODEL_NAME,
        prompt="A beautiful landscape with mountains and a river",
        output_dir=IMAGES_DIR,
        device="cpu"
    )

