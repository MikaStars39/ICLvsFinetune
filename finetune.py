import torch
import os
from transformers import (
    GPTNeoForCausalLM, 
    GPTNeoConfig, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from torch.distributed import barrier
from functools import partial

def main():

    model_name = "/home/qingyu_yin/model/llama-2-7B-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_format(tokenizer, examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs

    rank = int(os.environ.get('RANK', -1))

    if rank > 0:
        barrier()

    dataset = load_dataset("json", data_files="data/ft_data.json", split="train").shuffle(seed=42)
    dataset = dataset.map(partial(tokenize_and_format, tokenizer), batched=True, num_proc=16)

    if rank == 0:
        barrier()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model.config.use_cache = False # gradient checkpoint

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=100,
        weight_decay=0,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=1e-5,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=True,  # If you have a GPU with Tensor Cores
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # 开始训练
    trainer.train()

    # # 保存模型和分词器
    # trainer.save_model("./gpt-neo-1.4B-finetuned-wikitext-103")
    # tokenizer.save_pretrained("./gpt-neo-1.4B-finetuned-wikitext-103")

if __name__ == "__main__":
    main()