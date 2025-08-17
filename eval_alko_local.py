import math, os, glob, torch
import pyarrow as pa, pyarrow.ipc as pa_ipc
from datasets import load_from_disk, load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoModelForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
import alko  # register custom model

CKPT = r"01-pretraining-pipeline/results/checkpoints/run_alko_big"
VAL  = r"01-pretraining-pipeline/data/processed/val_big.arrow"
SPM  = r"01-pretraining-pipeline/results/tokenizer/spm.model"

def load_val_dataset(val_dir: str):
    # 1) try normal path (datasets v3)
    try:
        return load_from_disk(val_dir)
    except Exception as e:
        print("[eval] load_from_disk failed ->", e)
        
        # Try to fix the old 'List' feature type issue
        if "Feature type 'List' not found" in str(e):
            print("[eval] attempting to fix 'List' feature type issue...")
            try:
                # Look for dataset_info.json and fix the feature type
                info_file = os.path.join(val_dir, "dataset_info.json")
                if os.path.exists(info_file):
                    import json
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    print(f"[eval] original features: {info.get('features', 'Not found')}")
                    
                    # Replace 'List' with 'Sequence' in features
                    if 'features' in info:
                        features_str = json.dumps(info['features'])
                        if '"List"' in features_str:
                            features_str = features_str.replace('"List"', '"Sequence"')
                            info['features'] = json.loads(features_str)
                            
                            # Write back the fixed info
                            with open(info_file, 'w') as f:
                                json.dump(info, f, indent=2)
                            
                            print(f"[eval] fixed features: {info['features']}")
                            print("[eval] fixed dataset_info.json, trying to load again...")
                            return load_from_disk(val_dir)
                        else:
                            print("[eval] no 'List' found in features, issue might be elsewhere")
                    else:
                        print("[eval] no features found in dataset_info.json")
                else:
                    print("[eval] dataset_info.json not found")
            except Exception as fix_e:
                print(f"[eval] feature type fix failed -> {fix_e}")

    # 2) try parquet fallback (if your packer kept shards)
    pq_dir = val_dir + "_parquet"
    pq_files = glob.glob(os.path.join(pq_dir, "*.parquet"))
    if pq_files:
        print("[eval] loading parquet shards fallback")
        try:
            # Try to load parquet with explicit features to avoid 'List' type issue
            return load_dataset("parquet", data_files=pq_files, split="train", features=Features({"input_ids": Sequence(Value("int32"))}))
        except Exception as e:
            print(f"[eval] parquet loading failed -> {e}")
            print("[eval] trying to read parquet files directly...")
            
            # Try to read parquet files directly with pandas
            try:
                import pandas as pd
                all_rows = []
                for f in sorted(pq_files):
                    try:
                        df = pd.read_parquet(f)
                        if 'input_ids' in df.columns:
                            all_rows.extend(df['input_ids'].tolist())
                    except Exception as read_e:
                        print(f"[eval] failed to read parquet {f}: {read_e}")
                        continue
                
                if all_rows:
                    print(f"[eval] loaded {len(all_rows)} sequences from parquet files")
                    feats = Features({"input_ids": Sequence(Value("int32"))})
                    return Dataset.from_dict({"input_ids": all_rows}, features=feats)
            except ImportError:
                print("[eval] pandas not available, skipping parquet direct read")
            except Exception as pd_e:
                print(f"[eval] pandas parquet read failed -> {pd_e}")

    # 3) last resort: read arrow file(s) directly and rebuild features explicitly
    print("[eval] reading raw .arrow files and rebuilding dataset")
    arrow_files = glob.glob(os.path.join(val_dir, "data", "*.arrow")) or glob.glob(os.path.join(val_dir, "*.arrow"))
    assert arrow_files, f"No .arrow files found under {val_dir}"
    
    all_rows = []
    for f in sorted(arrow_files):
        try:
            with open(f, "rb") as fh:
                rb = pa_ipc.open_file(fh)
                tbl = rb.read_all()
            col = tbl.column("input_ids")  # list<int32>
            all_rows.extend(col.to_pylist())  # list[list[int]]
        except Exception as e:
            print(f"[eval] failed to read {f}: {e}")
            continue
    
    if not all_rows:
        raise ValueError("No data could be loaded from any source")
    
    print(f"[eval] loaded {len(all_rows)} sequences from arrow files")
    
    # Use the correct feature type for current datasets version
    feats = Features({"input_ids": Sequence(Value("int32"))})
    return Dataset.from_dict({"input_ids": all_rows}, features=feats)

def main():
    # tokenizer
    try:
        tok = LlamaTokenizer.from_pretrained(CKPT)
        print("Loaded tokenizer from checkpoint")
    except Exception:
        tok = LlamaTokenizer(vocab_file=SPM)
        print("Loaded tokenizer from SPM")
    if tok.eos_token is None: tok.add_special_tokens({"eos_token":"</s>","bos_token":"<s>"})
    if tok.pad_token is None: tok.add_special_tokens({"pad_token":"[PAD]"})

    # data
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

    # collator
    class FixedLenCollator:
        def __call__(self, feats):
            x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
            return {"input_ids": x, "labels": x.clone()}
    collator = FixedLenCollator()

    # eval perplexity
    args = TrainingArguments(output_dir="tmp_eval", per_device_eval_batch_size=1,
                             dataloader_num_workers=0, report_to=[])
    trainer = Trainer(model=model, args=args, data_collator=collator)
    metrics = trainer.evaluate(eval_dataset=ds)
    ppl = math.exp(metrics["eval_loss"])
    print(f"Eval loss: {metrics['eval_loss']:.4f}\nPerplexity: {ppl:.2f}")

    # sample
    prompt = "Hello, how are you? Can you tell me "
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=200, min_new_tokens=60,
        do_sample=True, temperature=0.9, top_p=0.95, top_k=50,
        repetition_penalty=1.1, no_repeat_ngram_size=3,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    print("\n--- SAMPLE ---")
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()