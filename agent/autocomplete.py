"""Fill-in-the-middle (FIM) autocomplete (Article 6, Part 1 / Step 8).

The cursor sits between a prefix (code above) and a suffix (code below); the
model fills the gap. Qwen2.5-Coder ships FIM special tokens, so we wrap the
request in them and call the legacy completions endpoint.

IMPORTANT: FIM needs the *base* model, not the instruct one. The instruct tag
(``qwen2.5-coder:1.5b``) applies a chat template and answers in prose instead of
filling the gap; ``qwen2.5-coder:1.5b-base`` does raw fill-in-the-middle.
"""
from __future__ import annotations

# Qwen FIM tokens; the exact names vary by model family.
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"
FIM_STOP = ["<|fim_pad|>", "<|file_sep|>", "<|endoftext|>"]

DEFAULT_FIM_MODEL = "qwen2.5-coder:1.5b-base"


def fim_complete(client, prefix: str, suffix: str,
                 model: str = DEFAULT_FIM_MODEL, max_tokens: int = 64,
                 single_line: bool = True) -> str:
    """Return the model's completion for the gap between prefix and suffix.

    ``single_line=True`` stops at the first newline, which is the common
    inline-completion case (finish the current line).
    """
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    stop = list(FIM_STOP) + (["\n"] if single_line else [])
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        stop=stop,
        temperature=0.1,
    )
    return response.choices[0].text
