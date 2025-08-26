import torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = r"01-pretraining-pipeline\results\checkpoints\run_llama_sft_mix_v4"
PROMPTS = [
    "Summarize: Why is gradient clipping used in training?",
    "Write Python code to reverse a linked list, with docstring and tests.",
    "Translate to Ukrainian: 'Large language models are useful for coding.'",
    "Step-by-step: how to create a venv and install PyTorch on Windows?",
    "Given a list of numbers, return a dict with mean/median/std in Python.",
    "You are a helpful assistant. Answer in JSON with keys {answer, steps} about: how to sort a dict by value?",
    "Refactor this function to be async-safe (explain changes):\n\ndef fetch_all(urls): ...",
    "Explain KV-cache in 3 bullet points, <=60 words total.",
]

BOUNDARY = "User: {}\nAssistant:\n"
GEN_KW = dict(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.05,
)

def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16, device_map="auto").eval()

    for q in random.sample(PROMPTS, k=min(8, len(PROMPTS))):
        prompt = BOUNDARY.format(q)
        ids = tok([prompt], return_tensors="pt").to(model.device)
        out = model.generate(**ids, **dict(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True))
        print("="*80, "\nQ:", q, "\nA:", tok.decode(out[0], skip_special_tokens=True).split("Assistant:\n",1)[-1].strip(), "\n")

if __name__ == "__main__":
    main()