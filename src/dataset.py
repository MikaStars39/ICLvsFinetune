from datasets import load_dataset

data_name_or_path = "../data/code.json"
dataset = load_dataset("json", data_files=data_name_or_path, split="train")
print(dataset[0]["code"])
