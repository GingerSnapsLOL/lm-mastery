from datasets import load_dataset

dolly  = load_dataset("databricks/databricks-dolly-15k", split="train")
ultra  = load_dataset("HuggingFaceH4/ultrachat_200k", split="train")
oasst  = load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")
hermes = load_dataset("teknium/OpenHermes-2.5", split="train")
prefs  = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train")

print(len(dolly), len(ultra), len(oasst), len(hermes), len(prefs))
