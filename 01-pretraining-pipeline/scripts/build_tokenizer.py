# python 01-pretraining-pipeline\scripts\build_tokenizer.py ^
#   --input_glob "01-pretraining-pipeline\data\raw\*.txt" ^
#   --vocab_size 32000 --model_type bpe ^
#   --output "01-pretraining-pipeline\results\tokenizer"



import argparse, os, glob
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--input_glob", required=True, help="Glob for text files")
parser.add_argument("--vocab_size", type=int, default=32000)
parser.add_argument("--model_type", default="bpe", choices=["bpe","unigram"])
parser.add_argument("--output", required=True, help="Output dir for tokenizer files")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

train_file = os.path.join(args.output, "spm_corpus.txt")
with open(train_file, "w", encoding="utf-8") as out:
    import glob as _glob
    for path in _glob.glob(args.input_glob):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line.strip() + "\n")

spm.SentencePieceTrainer.Train(
    input=train_file,
    model_prefix=os.path.join(args.output, "spm"),
    vocab_size=args.vocab_size,
    model_type=args.model_type,
    character_coverage=0.9995,
    input_sentence_size=2000000,
    shuffle_input_sentence=True
)

print(f"Tokenizer written to {args.output}. Files: spm.model, spm.vocab")
