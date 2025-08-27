\
import os, glob, json, argparse
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=50000)
    ap.add_argument("--model_prefix", default="alko2_spm")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")
    os.makedirs(args.out_dir, exist_ok=True)
    lst = os.path.join(args.out_dir, "spm_input_list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write("\\n".join(files))

    prefix = os.path.join(args.out_dir, args.model_prefix)
    spm.SentencePieceTrainer.Train(
        input=lst,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity"
    )

    src = prefix + ".model"
    dst = os.path.join(args.out_dir, "tokenizer.model")
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(fin.read())

    tok_cfg = {
        "model_max_length": 2048,
        "bos_token": {"id": 1, "content": "<s>"},
        "eos_token": {"id": 2, "content": "</s>"},
        "unk_token": {"id": 0, "content": "<unk>"},
        "pad_token": {"id": 2, "content": "</s>"},
        "clean_up_tokenization_spaces": False,
    }
    with open(os.path.join(args.out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)
    specials = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "</s>"}
    with open(os.path.join(args.out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(specials, f, indent=2)

    print("Tokenizer written to:", args.out_dir)

if __name__ == "__main__":
    main()
