# eval_llama_local.py
import os, glob, math, torch, json
import pyarrow as pa, pyarrow.ipc as pa_ipc
from datasets import load_from_disk, load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoModelForCausalLM, LlamaTokenizer, TrainingArguments, Trainer

CKPT = r"01-pretraining-pipeline/results/checkpoints/run_llama_baseline_109M"
VAL  = r"01-pretraining-pipeline/data/processed/val_big.arrow"
SPM  = r"01-pretraining-pipeline/results/tokenizer/spm.model"

def load_val_dataset(val_dir: str):
    # 1) normal (v3) path
    try:
        return load_from_disk(val_dir)
    except Exception as e:
        print("[eval] load_from_disk failed ->", e)

    # 2) parquet fallback (if you kept shards)
    pq_dir = val_dir + "_parquet"
    pq_files = glob.glob(os.path.join(pq_dir, "*.parquet"))
    if pq_files:
        print("[eval] loading parquet shards fallback")
        return load_dataset(
            "parquet", data_files=pq_files, split="train",
            features=Features({"input_ids": Sequence(Value("int32"))})
        )

    # 3) raw Arrow -> rebuild Dataset with explicit features
    print("[eval] reading raw .arrow files and rebuilding dataset")
    arrow_files = (glob.glob(os.path.join(val_dir, "data", "*.arrow"))
                   or glob.glob(os.path.join(val_dir, "*.arrow")))
    assert arrow_files, f"No .arrow files found under {val_dir}"
    all_rows = []
    for f in sorted(arrow_files):
        with open(f, "rb") as fh:
            rb = pa_ipc.open_file(fh)
            tbl = rb.read_all()
        all_rows.extend(tbl.column("input_ids").to_pylist())
    feats = Features({"input_ids": Sequence(Value("int32"))})
    return Dataset.from_dict({"input_ids": all_rows}, features=feats)

def main():
    print("[datasets] version check:", __import__("datasets").__version__)
    # tokenizer
    try:
        tok = LlamaTokenizer.from_pretrained(CKPT)
        print("Loaded tokenizer from checkpoint")
    except Exception:
        tok = LlamaTokenizer(vocab_file=SPM)
        print("Loaded tokenizer from SPM")
    if tok.eos_token is None: tok.add_special_tokens({"eos_token":"</s>","bos_token":"<s>"})
    if tok.pad_token is None: tok.add_special_tokens({"pad_token":"[PAD]"})

    ds = load_val_dataset(VAL)
    print("[eval] rows:", len(ds), "| example len:", len(ds[0]["input_ids"]))

    # model
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=dtype)
    if torch.cuda.is_available(): model.cuda()
    model.config.use_cache = False
    if model.get_input_embeddings().num_embeddings != len(tok):
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id

    class Coll:
        def __call__(self, feats):
            x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
            return {"input_ids": x, "labels": x.clone()}

    args = TrainingArguments(
        output_dir="tmp_eval_llama",
        per_device_eval_batch_size=1,
        dataloader_num_workers=0,
        report_to=[]
    )
    metrics = Trainer(model=model, args=args, data_collator=Coll()).evaluate(eval_dataset=ds)
    print(f"Eval loss: {metrics['eval_loss']:.4f}\nPerplexity: {math.exp(metrics['eval_loss']):.2f}")

    # sample
    prompt = "Hello, how are you? Can you tell me "
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, min_new_tokens=60, do_sample=True,
                         temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.1,
                         no_repeat_ngram_size=3, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    print("\n--- SAMPLE ---\n", tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
