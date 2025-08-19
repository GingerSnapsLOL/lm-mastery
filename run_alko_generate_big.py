import os, torch
import alko  # registers AlkoConfig/AlkoForCausalLM with Auto*
from transformers import AutoModelForCausalLM, LlamaTokenizer

CKPT = r"01-pretraining-pipeline/results/checkpoints/run_alko_big"
SPM  = r"01-pretraining-pipeline/results/tokenizer/spm.model"

def load_tokenizer():
    try:
        tok = LlamaTokenizer.from_pretrained(CKPT)
        print("Loaded tokenizer from checkpoint")
    except Exception:
        tok = LlamaTokenizer(vocab_file=SPM)
        print("Loaded tokenizer from SPM")
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token":"</s>", "bos_token":"<s>"})
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"[PAD]"})
    return tok

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = load_tokenizer()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=dtype)
    model.config.use_cache = False

    # ensure PAD id is set and distinct; resize if we added a new token
    if model.config.pad_token_id is None or tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token":"[PAD]"})
    if model.get_input_embeddings().num_embeddings != len(tok):
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id

    if torch.cuda.is_available():
        model.to("cuda")

    prompt = "Hello, how are you? Can you tell me "
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    # DEBUG: forbid EOS to avoid silent outputs (remove after sanity check)
    forbid_eos = True
    bad = [[tok.eos_token_id]] if forbid_eos and tok.eos_token_id is not None else None

    out = model.generate(
        **inputs,
        max_new_tokens=256,
        min_new_tokens=64,
        do_sample=True, temperature=0.9, top_p=0.95, top_k=50,
        repetition_penalty=1.1, no_repeat_ngram_size=3,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        bad_words_ids=bad,                # <— this replaces SuppressTokens
    )

    print(tok.decode(out[0], skip_special_tokens=True))
    print("-" * 100)
    print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")

    # Optional: inspect next-token probs to see if EOS dominates
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze()
        topk = torch.topk(probs, 10)
        pairs = [(tok.decode([i]).replace("\n","\\n"), float(p)) for i,p in zip(topk.indices, topk.values)]
        print("Top-10 next tokens:", pairs)

if __name__ == "__main__":
    main()
