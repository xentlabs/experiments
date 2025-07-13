import json
import logging
import re
from typing import Any, List

import torch
import torch.nn.functional as F

from shared.types import ChatMessage

BIG_MODELS = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-30B-A3B",
    # DeepSeek v2
    # "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    # DeepSeek v3
    "deepseek-ai/DeepSeek-V3",
    # Llama 4
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # Llama 3
    "meta-llama/Meta-Llama-3-70B-Instruct",
    # Mistral 3 Small
    "mistralai/Mistral-Small-24B-Instruct-2501",
    # Gemma 3
    "google/gemma-3-27b-it",
]

ALL_MODELS = [
    "Qwen/Qwen2-72B-Instruct",
    # Qwen 2.5
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    # Qwen 3
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
    # DeepSeek v2
    # "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    # DeepSeek v3
    "deepseek-ai/DeepSeek-V3",
    # Llama 4
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # Llama 3
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    # Mistral 3 Small
    "mistralai/Mistral-Small-24B-Instruct-2501",
    # Gemma 3
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
]

ALL_BASE_MODELS = [
    # Qwen 2
    "Qwen/Qwen2-72B",
    # Qwen 2.5
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-14B",
    # Qwen 3
    # Not all qwen3 base models are released yet: https://huggingface.co/Qwen/Qwen3-235B-A22B/discussions/17
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-0.6B-Base",
    # DeepSeek v2
    # "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V2-Lite",
    # DeepSeek v3
    # "deepseek-ai/DeepSeek-V3-Base",
    "deepseek-ai/DeepSeek-V3-Base-Override",
    # Llama 4
    "meta-llama/Llama-4-Scout-17B-16E",
    "meta-llama/Llama-4-Maverick-17B-128E",
    # Llama 3
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-8B",
    # Mistral 3 Small
    "mistralai/Mistral-Small-24B-Base-2501",
    # Gemma 3
    "google/gemma-3-27b-pt",
    "google/gemma-3-12b-pt",
    "google/gemma-3-4b-pt",
    "google/gemma-3-1b-pt",
    # SmolLM
    "HuggingFaceTB/SmolLM-1.7B",
    "HuggingFaceTB/SmolLM-360M",
    "HuggingFaceTB/SmolLM-135M",
]


def tokenize(string: str, tokenizer: Any) -> Any:
    return tokenizer(string, return_tensors="pt")


def tokenize_to_strings(string: str, tokenizer: Any) -> List[str]:
    tokenized_inputs = tokenizer(string, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"].squeeze(0)
    token_strings = [tokenizer.decode([token_id]) for token_id in input_ids]
    return token_strings


def comp_logits(tokenized_inputs: Any, model: Any) -> Any:
    target_device = model.get_input_embeddings().weight.device
    prepared_inputs = {}
    for key, value in tokenized_inputs.items():
        if isinstance(value, torch.Tensor) and (value.device != target_device):
            prepared_inputs[key] = value.to(device=target_device)
        else:
            prepared_inputs[key] = value

    result = model(**prepared_inputs, return_dict=True)

    # result = model(**tokenized_inputs, return_dict=True)
    return result.logits.squeeze(0)


def per_token_xent(string: str, model: Any, tokenizer: Any):
    cpu_tokenized_inputs = tokenize(string, tokenizer)
    logits = comp_logits(cpu_tokenized_inputs, model)

    cpu_target_tokens = cpu_tokenized_inputs["input_ids"].squeeze(0)[1:]
    target_tokens = cpu_target_tokens.to(logits.device)

    pred_logits = logits[:-1, :]
    score = F.cross_entropy(pred_logits, target_tokens, reduction="none")

    token_ids = cpu_target_tokens.tolist()
    token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]

    scores_python = [float(x) for x in score.cpu().numpy()]

    return list(zip(token_strings, scores_python))


def xent(string: str, model: Any, tokenizer: Any) -> float:
    cpu_tokenized_inputs = tokenize(string, tokenizer)
    logits = comp_logits(cpu_tokenized_inputs, model)

    # Target tokens are derived from input_ids from the CPU dictionary
    cpu_target_tokens = cpu_tokenized_inputs["input_ids"].squeeze(0)[1:]
    target_tokens = cpu_target_tokens.to(logits.device)

    pred_logits = logits[:-1, :]
    score = F.cross_entropy(pred_logits, target_tokens, reduction="sum")
    return score.item()


def generate_chat_response(
    conversation_history: List[ChatMessage],
    model: Any,
    tokenizer: Any,
    **generation_kwargs,
) -> tuple[str, List[ChatMessage]]:
    model.eval()

    logging.info("Applying chat template")
    prompt_input_ids = tokenizer.apply_chat_template(
        conversation_history,
        add_generation_prompt=True,  # Tells the model it's its turn
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(prompt_input_ids).to(model.device)

    logging.info("Generating")
    logging.info(
        f"Full conversation history: {json.dumps(conversation_history, indent=4)}"
    )
    generated_ids = model.generate(
        prompt_input_ids, attention_mask=attention_mask, **generation_kwargs
    )

    response_ids = generated_ids[:, prompt_input_ids.shape[-1] :]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    updated_history = conversation_history + [
        {"role": "assistant", "content": response_text}
    ]

    return response_text.strip(), updated_history


def remove_think_content(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
