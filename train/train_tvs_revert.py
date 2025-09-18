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
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from prompts import (
    BOV_TOKEN,
    EOV_TOKEN,
    CON_TOKEN,
    DELIMITER,
    QWEN_CHAT_TEMPLATE,
    SYSTEM_PROMPT_FOR_VERBALIZER,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class VerbalizerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bov_token = BOV_TOKEN
        self.eov_token = EOV_TOKEN
        self.con_token = CON_TOKEN
        self.delimiter = DELIMITER
        self.bov_token_id = self.processing_class.convert_tokens_to_ids(self.bov_token)
        self.eov_token_id = self.processing_class.convert_tokens_to_ids(self.eov_token)
        self.con_token_id = self.processing_class.convert_tokens_to_ids(self.con_token)
        self.generation_kwargs = {
            "use_cache": True,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "return_dict_in_generate": True,
        }

class DataManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        bov_token: str,
        eov_token: str,
        con_token: str,
        delimiter: str,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.bov_token = bov_token
        self.eov_token = eov_token
        self.con_token = con_token
        self.bov_token_id = self.tokenizer.convert_tokens_to_ids(bov_token)
        self.eov_token_id = self.tokenizer.convert_tokens_to_ids(eov_token)
        self.con_token_id = self.tokenizer.convert_tokens_to_ids(con_token)
        self.delimiter = delimiter
        self.delimiter_token_dict = self.get_all_delimiter_token_ids()
        self.max_length = max_length

    def get_all_delimiter_token_ids(self):
        delimiter_token_dict = {self.con_token_id: self.con_token}
        for token, token_id in self.tokenizer.get_vocab().items():
            if self.tokenizer.decode([token_id]).endswith(self.delimiter):
                delimiter_token_dict[token_id] = token

        return delimiter_token_dict

    def text_to_chunks(self, text: str):
        text = re.sub("\n+", "\n", text)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        curr_chunk = []
        for token_id in token_ids:
            curr_chunk.append(token_id)
            if token_id in self.delimiter_token_dict.keys():
                chunks.append(curr_chunk)
                curr_chunk = []

        if curr_chunk:
            chunks.append(curr_chunk)

        return chunks

    def preprocess_train_dataset(self, example):
        scatter = example["scatter"]
        scatter = re.sub("\n+", "\n", scatter)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_VERBALIZER},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": scatter},
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            padding="max_length",
            truncation=True,
            return_dict=True,
            max_length=self.max_length,
            return_assistant_tokens_mask=True,
            chat_template=QWEN_CHAT_TEMPLATE,
            add_generation_prompt=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        assistant_mask = tokenized["assistant_masks"]

        # Set labels
        labels = [-100] * len(input_ids)

        is_in_bov_eov = False
        for i, token_id in enumerate(input_ids[:-1]):
            next_token_id = input_ids[i + 1]
            if assistant_mask[i] == 0:
                continue

            if next_token_id == self.bov_token_id:
                is_in_bov_eov = True
            elif token_id == self.eov_token_id:
                is_in_bov_eov = False

            if is_in_bov_eov:
                labels[i + 1] = next_token_id
            elif token_id in self.delimiter_token_dict.keys():
                labels[i + 1] = self.con_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-3B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for training."},
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="all",
        metadata={"help": "The name of the dataset to use."},
    )

    max_length: int = field(
        default=4096,
        metadata={"help": "The maximum length of the input sequence."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if "llama" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Add special tokens
    logger.info("Adding special tokens...")
    special_tokens = [BOV_TOKEN, EOV_TOKEN, CON_TOKEN]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Load data
    if data_args.dataset_name == "all":
        datasets = [
            GSM8kDataset.load_train_dataset_for_tvs_revert(),
            MHQADataset.load_train_dataset_for_tvs_revert(),
        ]
    elif data_args.dataset_name == "gsm8k":
        datasets = [
            GSM8kDataset.load_train_dataset_for_tvs_revert(),
        ]
    elif data_args.dataset_name == "2wikimultihop":
        datasets = [
            MHQADataset.load_train_dataset_for_tvs_revert(),
        ]
    else:
        raise ValueError(f"Unknown dataset name: {data_args.dataset_name}")

    train_dataset = concatenate_datasets([dataset for dataset in datasets])
    train_dataset = train_dataset.shuffle(seed=training_args.seed)

    data_preprocessor = DataManager(
        tokenizer=tokenizer,
        bov_token=BOV_TOKEN,
        eov_token=EOV_TOKEN,
        con_token=CON_TOKEN,
        delimiter=DELIMITER,
        max_length=data_args.max_length,
    )

    with PartialState().main_process_first():
        train_dataset = train_dataset.map(
            data_preprocessor.preprocess_train_dataset,
            remove_columns=train_dataset.column_names,
            desc="Preprocessing train dataset",
            num_proc=24,
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(training_args.device)

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    if model_args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"]
        )

        model = get_peft_model(model, peft_config)

    # Train model
    trainer = VerbalizerTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info("Training the model...")
    trainer.train()

    # Save Models
    logger.info("Saving model and tokenizer...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)