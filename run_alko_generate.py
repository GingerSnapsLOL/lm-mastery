import torch, os
import alko  # registers AlkoConfig/AlkoForCausalLM with Auto*
from transformers import AutoModelForCausalLM, LlamaTokenizer

CKPT = r"01-pretraining-pipeline/results/checkpoints/run_alko_wt103"

@torch.no_grad()
def simple_generate(model, tok, prompt, max_new_tokens=100, temperature=1.0, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :]  # last-step logits
        logits = logits / max(temperature, 1e-6)

        # nucleus (top-p) sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum > top_p
        # ensure at least one token stays
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_id = sorted_idx.gather(1, torch.multinomial(sorted_probs, 1))

        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0], skip_special_tokens=True)







# Tokenizer (load from checkpoint; fallback to SPM if needed)
try:
    tok = LlamaTokenizer.from_pretrained(CKPT)
    print("Loaded tokenizer from checkpoint")
except Exception:
    tok = LlamaTokenizer(vocab_file=r"01-pretraining-pipeline/results/tokenizer/spm.model")
    print("Loaded tokenizer from SPM")

# Add special tokens if not present
if tok.eos_token is None:
    tok.add_special_tokens({"eos_token":"</s>","bos_token":"<s>"})
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Load model
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=dtype)
model.config.use_cache = False 

if torch.cuda.is_available():
    model.to("cuda")

prompt = "You are AlkoForCausalLM. In two sentences, what is a transformer?"
inputs = tok(prompt, return_tensors="pt").to(model.device)



try:

    torch.manual_seed(42)

    out = model.generate(
        **inputs,
        max_new_tokens=256,          # more tokens
        min_new_tokens=64,           # force at least this many
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))
    print("-"*100)  

    print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")
except Exception as e:
    print("-"*100)
    print(e)
    print("-"*100)
    print(simple_generate(model, tok, 
    "You are AlkoForCausalLM. In one sentence, what is a transformer?", 120, 0.9, 0.95))
    print("-"*100)


