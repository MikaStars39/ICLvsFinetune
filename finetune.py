import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from functools import partial

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.distributed import barrier

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        example["text"],
        truncation=True,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Now loading model from {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # process data
    rank = int(os.environ.get('RANK', -1))
    print("Now processing data")
    if rank > 0:
        barrier()
    dataset = load_dataset("json", data_files=data_args.data_path, split="train").shuffle(42)
    dataset = dataset.map(partial(tokenize_fn, tokenizer), batched=True, num_proc=32)
    if rank == 0:
        barrier()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # train
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()