"""Shared training/eval plumbing: chat rendering, QLoRA setup, generation."""

from __future__ import annotations

import json
from pathlib import Path

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TINY_SMOKE_MODEL = "hf-internal-testing/tiny-random-MistralForCausalLM"


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def render_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Render messages to text. Uses the tokenizer chat template when present;
    falls back to a plain format for tiny test models without one."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    prompt = f"### System\n{system}\n### User\n{user}\n### Assistant\n"
    if add_generation_prompt:
        return prompt
    assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    return prompt + assistant


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def build_model_and_tokenizer(
    model_name: str = DEFAULT_BASE_MODEL,
    adapter: str | None = None,
    quantize: str = "auto",  # "auto" | "nf4" | "none"
    device: str = "auto",
):
    """Load (optionally QLoRA-quantized) model + tokenizer, mounting a LoRA
    adapter when provided. Lazy imports keep non-ML paths import-free."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_4bit = False
    if quantize == "nf4":
        use_4bit = True
    elif quantize == "auto":
        use_4bit = torch.cuda.is_available() and _bnb_available()

    kwargs: dict = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not use_4bit:
        model = model.to(device)
    model.eval()

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def make_lora_config(r: int = 16, lora_alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
    )


def tokenize_chat_example(tokenizer, messages: list[dict], max_len: int = 2048):
    """Tokenize one chat example with prompt tokens masked out of the loss.

    Returns dict(input_ids, labels, attention_mask) as plain lists.
    """
    prompt_text = render_chat(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = render_chat(tokenizer, messages)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if len(full_ids) > max_len:  # keep the tail (diff + answer) on overflow
        full_ids = full_ids[-max_len:]
        labels = labels[-max_len:]
    attention = [1] * len(full_ids)
    return {"input_ids": full_ids, "labels": labels, "attention_mask": attention}


def generate_texts(
    model,
    tokenizer,
    prompts: list[list[dict]],
    max_new_tokens: int = 96,
    batch_size: int = 4,
) -> list[str]:
    """Greedy batch generation from chat messages."""
    import torch

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out: list[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            texts = [render_chat(tokenizer, m, add_generation_prompt=True) for m in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j in range(len(batch)):
                new_tokens = gen[j][enc["input_ids"].shape[1]:]
                out.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    return out
