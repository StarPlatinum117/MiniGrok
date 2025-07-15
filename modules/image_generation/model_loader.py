from diffusers import StableDiffusionPipeline
import torch


def load_model(
        *,
        model_name: str,
        device: str = "cpu",
) -> StableDiffusionPipeline:
    """
    Load the Stable Diffusion model for image generation.

    Args:
        model_name: The name of the model to load.
        device: The device to load the model on (default is "cpu").

    Returns:
        StableDiffusionPipeline: The loaded Stable Diffusion pipeline.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32, use_safetensors=True)
    pipe.to(device)
    pipe.safety_checker = None

    return pipe
