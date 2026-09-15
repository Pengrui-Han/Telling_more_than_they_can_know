"""Load the model and run one greedy chat completion."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model(name: str = DEFAULT_MODEL, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="auto")
    model.eval()
    return tokenizer, model


@torch.no_grad()
def chat(tokenizer, model, user_message: str | list[dict[str, str]], max_new_tokens: int = 200) -> str:
    """Chat with a user string or explicit message history; returns the greedy reply.

    repetition_penalty is set explicitly because Qwen2.5's generation_config.json ships
    with 1.05, which do_sample=False alone would silently keep.
    """
    messages = ([{"role": "user", "content": user_message}]
                if isinstance(user_message, str) else user_message)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # The chat template already supplies BOS/EOS and other control tokens.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.0, num_beams=1, num_return_sequences=1,
                         use_cache=True, return_dict_in_generate=False)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
