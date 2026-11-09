"""Loading a model with vLLM, with clear errors for common problems.

Handles both kinds of checkpoint:
* merged weights (config.json + *.safetensors) -> loaded directly;
* LoRA adapters (adapter_config.json)          -> the base model is loaded
  with LoRA enabled and the adapter is applied at generation time.
"""

import json
import os
import sys


def fail(msg):
    sys.exit(f"\nERROR: {msg}")


def check_runtime():
    """Import torch and vLLM and make sure a CUDA GPU is available."""
    try:
        import torch
    except ImportError:
        fail("PyTorch is not installed. Run: pip install -r requirements.txt")
    try:
        import vllm  # noqa: F401
    except ImportError:
        fail("vLLM is not installed. Run: pip install -r requirements.txt "
             "(vLLM needs Linux and an NVIDIA GPU).")
    if not torch.cuda.is_available():
        fail("No CUDA GPU found. vLLM inference needs an NVIDIA GPU "
             "(about 18 GB of free memory for an 8B model in 16-bit).")
    return torch


def _list_files(model):
    """Files of a local folder or a Hugging Face repo (empty list if unknown)."""
    if os.path.isdir(model):
        return os.listdir(model)
    try:
        from huggingface_hub import list_repo_files
        return list_repo_files(model)
    except Exception:
        return []


def inspect_checkpoint(model):
    """Return ('merged'|'adapter'|'unknown', base_model_or_None)."""
    files = _list_files(model)
    if "adapter_config.json" in files and not any(f.endswith("config.json") and f != "adapter_config.json"
                                                  and "/" not in f for f in files):
        base = None
        try:
            if os.path.isdir(model):
                path = os.path.join(model, "adapter_config.json")
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(model, "adapter_config.json")
            with open(path, encoding="utf-8") as f:
                base = json.load(f).get("base_model_name_or_path")
        except Exception:
            pass
        return "adapter", base
    if "config.json" in files:
        return "merged", None
    return "unknown", None


def load_llm(model, gpu_memory_utilization=0.80, max_model_len=8192, dtype="auto",
             base_model=None, tokenizer=None, **kwargs):
    """Create a vLLM engine. Returns (llm, lora_request_or_None)."""
    check_runtime()
    from vllm import LLM

    kind, adapter_base = inspect_checkpoint(model)
    lora_request = None
    engine_model = model
    engine_kwargs = dict(
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        **kwargs,
    )

    if kind == "adapter":
        engine_model = base_model or adapter_base
        if not engine_model:
            fail(f"{model} contains LoRA adapters but the base model is unknown; pass --base-model.")
        from vllm.lora.request import LoRARequest
        adapter_path = model
        if not os.path.isdir(model):
            from huggingface_hub import snapshot_download
            adapter_path = snapshot_download(model)
        print(f"{model} is a LoRA adapter; loading base model {engine_model} with LoRA enabled.")
        engine_kwargs.update(enable_lora=True, max_lora_rank=64)
        lora_request = LoRARequest("framellama", 1, adapter_path)

    try:
        llm = LLM(model=engine_model, tokenizer=tokenizer or engine_model, **engine_kwargs)
    except Exception as exc:  # turn the most common failures into readable messages
        text = str(exc).lower()
        if "out of memory" in text or "cuda oom" in text or "no available memory" in text \
                or "not enough" in text and "memory" in text:
            fail(f"Not enough GPU memory to load {engine_model}. Try a lower --max-model-len "
                 f"(now {max_model_len}), close other GPU jobs, or adjust "
                 f"--gpu-memory-utilization (now {gpu_memory_utilization}).\nDetails: {exc}")
        if "gated" in text or "401" in text or "403" in text or "access" in text and "restricted" in text:
            fail(f"Access to {engine_model} was refused. Llama models are gated: accept the license "
                 f"on Hugging Face and run `huggingface-cli login`.\nDetails: {exc}")
        raise
    return llm, lora_request


def generate(llm, prompts, sampling_params, lora_request=None):
    """llm.generate with an optional LoRA request."""
    if lora_request is not None:
        return llm.generate(prompts, sampling_params, lora_request=lora_request)
    return llm.generate(prompts, sampling_params)
