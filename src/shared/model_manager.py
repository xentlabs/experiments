import functools
import gc
import logging
import os
from typing import Any, Callable, TypeVar, cast

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is not None:
    login(token=HF_TOKEN)

TRUSTED_CODE_MODELS = [
    "deepseek-ai/DeepSeek-V2-Chat",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V2-Lite",
]

# Deepseek v3 override. Note that this requires cloning the repo and converting the model to bf16.
# Only required for A100 gpus - h100s are fine with the original model.
DSV3_NAME = "deepseek-ai/DeepSeek-V3-Base-Override"

# Type variables for the decorator
T = TypeVar("T", bound=Callable[..., Any])


def log_memory_usage(tag: str, device: torch.device):
    if device == "cuda":
        logging.info("[Decorator] CUDA Memory after cleanup:")
        for i in range(torch.cuda.device_count()):
            device_str = f"cuda:{i}"
            mem_allocated = torch.cuda.memory_allocated(device=device_str) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device=device_str) / 1024**2
            logging.info(
                f"  Device {device_str}: Allocated={mem_allocated:.2f}MB, Reserved={mem_reserved:.2f}MB"
            )
    elif device == "mps":
        try:
            mps_allocated = torch.mps.current_allocated_memory() / 1024**2
            logging.info(
                f"[Decorator] MPS Memory after cleanup: Current Allocated={mps_allocated:.2f}MB"
            )
        except AttributeError:
            logging.warning(
                "[Decorator] torch.mps.current_allocated_memory() not available."
            )

    mem_allocated = torch.cuda.memory_allocated(device=device) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(device=device) / 1024**2
    logging.info(
        f"{tag}: Allocated={mem_allocated:.2f}MB, Reserved={mem_reserved:.2f}MB"
    )


def with_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    hf_dir_path: str | None = None,
):
    """
    A decorator that loads a Hugging Face model and tokenizer,
    injects them into the decorated function, and ensures cleanup
    after the function completes.

    The decorated function must accept 'model' and 'tokenizer' parameters.

    Args:
        model_name: Name or path of the Hugging Face model
        device: PyTorch device to load the model onto
        hf_dir_path: Optional local directory containing models

    Returns:
        A decorator function
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            model = None
            tokenizer = None
            name_or_path = model_name
            if (hf_dir_path is not None) and os.path.exists(hf_dir_path):
                name_or_path = os.path.join(hf_dir_path, model_name)
            trusted_code = False
            if model_name in TRUSTED_CODE_MODELS:
                trusted_code = True
            if model_name == DSV3_NAME:
                name_or_path = "~/ubuntu/DeepSeek-V3-Base/bf16"
            logging.info(f"[Decorator] Acquiring model/tokenizer: {model_name}...")
            with torch.no_grad():
                try:
                    if "Llama-4" in model_name or "DeepSeek" in model_name:
                        model = AutoModelForCausalLM.from_pretrained(
                            name_or_path,
                            device_map="auto",
                            trust_remote_code=trusted_code,
                            torch_dtype=torch.bfloat16,
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            name_or_path,
                            device_map="auto",
                            trust_remote_code=trusted_code,
                        )

                    logging.info(f"DEVICE MAP: {model.hf_device_map}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        name_or_path,
                        trust_remote_code=trusted_code,
                    )
                    logging.info(
                        f"[Decorator] Acquired: {model_name} (Model ID: {id(model)})"
                    )

                    # Inject model and tokenizer into the decorated function
                    return func(*args, model=model, tokenizer=tokenizer, **kwargs)
                except Exception as e:
                    logging.exception(f"[Decorator] Error loading model/tokenizer: {e}")
                    raise
                finally:
                    # Release resources immediately after function execution
                    logging.info(
                        f"[Decorator] Releasing resources for {model_name} (Model ID: {id(model) if model else 'N/A'})..."
                    )

                    log_memory_usage("[Decorator] CUDA Memory before cleanup", device)

                    # Delete references to allow garbage collection
                    del model
                    del tokenizer

                    # Force garbage collection
                    collected = gc.collect()
                    logging.info(
                        f"[Decorator] Garbage collector collected {collected} objects."
                    )

                    # Clear device cache
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        logging.info("[Decorator] Cleared PyTorch CUDA cache.")
                    elif device.type == "mps":
                        try:
                            torch.mps.empty_cache()
                            logging.info("[Decorator] Cleared PyTorch MPS cache.")
                        except AttributeError:
                            logging.warning(
                                "[Decorator] torch.mps.empty_cache() not available in this PyTorch version."
                            )

                    log_memory_usage(
                        "[Decorator] CUDA Memory after cleanup",
                        device,
                    )

                    logging.info(
                        f"[Decorator] Finished releasing resources for {model_name}."
                    )

        return cast(T, wrapper)

    return decorator
