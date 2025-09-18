import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path
import re
from tqdm import tqdm
from datetime import datetime
import logging
from dataclasses import dataclass, field
from accelerate import PartialState

import torch
from datasets import concatenate_datasets
from data import GSM8kDataset, MHQADataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from prompts import SYSTEM_PROMPTS

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA."},
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="all",
        metadata={"help": "The name of the dataset to use."},
    )

    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"

    if "llama" in model_args.model_name_or_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # 2. Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).to(training_args.device)

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3. Setup LoRA
    if model_args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

    # Print Trainable Parameters on Rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("#" * 100)
        print(f"Total Parameters     : {total_params}")
        print(f"Trainable Parameters : {trainable_params}")
        print(f"Ratio                : {trainable_params / (1e-10 + total_params):.2%}")
        print("#" * 100)

    # 4. Load dataset
    if data_args.dataset_name == "all":
        datasets = [
            GSM8kDataset.load_train_dataset_for_tvs_seq(),
            MHQADataset.load_train_dataset_for_tvs_seq(),
        ]
    elif data_args.dataset_name == "gsm8k":
        datasets = [
            GSM8kDataset.load_train_dataset_for_tvs_seq(),
        ]
    elif data_args.dataset_name == "2wikimultihop":
        datasets = [
            MHQADataset.load_train_dataset_for_tvs_seq(),
        ]
    else:
        raise ValueError(f"Unknown dataset name: {data_args.dataset_name}")

    train_dataset = concatenate_datasets([dataset for dataset in datasets])
    train_dataset = train_dataset.shuffle(seed=training_args.seed)

    def preprocess_data(x):
        question = x["question"]
        verbalize = x["summarize"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["SFP"]},
            {"role": "user", "content": question},
            {"role": "assistant", "content": verbalize},
        ]

        chat_messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        user_messages = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        tokenized_messages = tokenizer(
            chat_messages, return_tensors="pt", padding=True, truncation=True, max_length=data_args.max_length
        )
        tokenized_user_messages = tokenizer(user_messages, return_tensors="pt")

        start_assistant_response_idx = len(tokenized_user_messages["input_ids"][0])

        input_ids = tokenized_messages["input_ids"].squeeze(0)
        attention_mask = tokenized_messages["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask the labels to -100
        # 1. Before the assistant response
        # 2. Pad tokens
        labels[:start_assistant_response_idx] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    with PartialState().main_process_first():
        train_dataset = train_dataset.map(
            preprocess_data,
            remove_columns=train_dataset.column_names,
            desc="Generate SFT dataset for Base LLM with summarized text",
            num_proc=16,
        )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
