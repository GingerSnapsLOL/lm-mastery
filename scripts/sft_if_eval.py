# sft_if_eval.py
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = r"01-pretraining-pipeline\results\checkpoints\run_llama_sft_mix_v4"
BOUNDARY = "User: {}\nAssistant:\n"
GEN = dict(max_new_tokens=256, temperature=0.2, do_sample=False)  # deterministic for grading

CASES = [
    dict(
        name="json_format",
        prompt="Answer in pure JSON with keys {answer, steps}. Topic: how to compute median in Python?",
        check=lambda s: _is_valid_json_with_keys(s, ["answer","steps"]),
    ),
    dict(
        name="bullets_3",
        prompt="List exactly 3 bullet points about KV cache in transformers. No extra text.",
        check=lambda s: _count_bullets(s)==3,
    ),
    dict(
        name="regex_email",
        prompt="Output a single line that matches this regex: ^[a-z]{3}\\d{2}@example\\.com$ . No explanation.",
        check=lambda s: re.fullmatch(r"[a-z]{3}\d{2}@example\.com", s.strip()) is not None,
    ),
    dict(
        name="python_function",
        prompt="Write a Python function `is_palindrome(s: str) -> bool` and nothing else.",
        check=lambda s: "def is_palindrome" in s and "return" in s,
    ),
    dict(
        name="word_limit",
        prompt="Explain gradient clipping in <= 40 words. No lists.",
        check=lambda s: len(s.split()) <= 40 and "-" not in s.split()[0],
    ),
]

def _is_valid_json_with_keys(txt, keys):
    try:
        # try to extract last JSON block
        m = re.search(r"\{[\s\S]*\}$", txt.strip())
        obj = json.loads(m.group(0) if m else txt)
        return all(k in obj for k in keys)
    except Exception:
        return False

def _count_bullets(s):
    return sum(1 for line in s.splitlines() if re.match(r"^\s*[-•*]\s+\S", line))

def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16, device_map="auto").eval()

    passed, total = 0, len(CASES)
    for c in CASES:
        prompt = BOUNDARY.format(c["prompt"])
        ids = tok([prompt], return_tensors="pt").to(model.device)
        out = model.generate(**ids, **GEN)
        ans = tok.decode(out[0], skip_special_tokens=True).split("Assistant:\n",1)[-1].strip()
        ok = c["check"](ans)
        print(f"[{c['name']}] PASS={ok}\n---\n{ans}\n")
        passed += int(ok)
    print(f"IF pass-rate: {passed}/{total} = {passed/total:.2%}")

if __name__ == "__main__":
    main()
