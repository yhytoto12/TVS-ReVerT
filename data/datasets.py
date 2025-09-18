from datasets import load_dataset, Dataset
import json
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompts import BOV_TOKEN, EOV_TOKEN


def preprocess_scatter_text_for_tvs_revert(text: str) -> str:
    # First find all ... <bov> ... <eov> pattern
    pattern = r"(.*?)<bov>(.*?)<eov>"
    matches = re.findall(pattern, text, re.DOTALL)

    new_text = ""
    for match in matches:
        think_content = match[0]
        speak_content = match[1]

        # strip for think_content
        think_content = think_content.strip()

        # Reformat
        new_text += f"{think_content}\n{BOV_TOKEN}{speak_content}{EOV_TOKEN}\n"

    # Find last match index
    if len(matches) == 0:
        return ""

    return new_text


def preprocess_scatter_text_for_tvs_seq(text: str) -> str:

    pattern = r"(.*?)<bov>(.*?)<eov>"
    matches = re.findall(pattern, text, re.DOTALL)

    # <bov>... <eov> ...
    think_text = ""
    speak_text = ""
    for match in matches:
        think_content = match[0]
        speak_content = match[1]

        # strip for think_content
        think_content = think_content.strip()

        think_text += think_content + "\n"
        speak_text += speak_content + " "

    # Reformat
    new_text = f"{think_text}\n{BOV_TOKEN}{speak_text}{EOV_TOKEN}\n"
    return new_text


def get_summary(text: str) -> str:
    pattern = r"(.*?)<bov>(.*?)<eov>"
    matches = re.findall(pattern, text, re.DOTALL)

    speak_contents = []
    for match in matches:
        think_content = match[0].strip()
        speak_content = match[1].strip()

        speak_contents.append(speak_content)

    summary = " ".join(speak_contents)
    return summary


class GSM8kDataset:
    dataset_name = "gsm8k"
    dataset_fname = "data/train_gsm8k.jsonl"
    dataset_repo = "yhytoto12/tvs-gsm8k"

    @staticmethod
    def load_train_dataset() -> Dataset:
        if os.path.exists(GSM8kDataset.dataset_fname):
            dataset = load_dataset("json", data_files={"train": GSM8kDataset.dataset_fname})["train"]
        else:
            dataset = load_dataset(GSM8kDataset.dataset_repo, split="train")
        return dataset

    @staticmethod
    def load_train_dataset_for_tvs_revert() -> Dataset:
        dataset = GSM8kDataset.load_train_dataset()

        dataset = dataset.map(
            lambda x: {
                "question": x["question"],
                "short_answer": x["short_answer"],
                "summarize": get_summary(x["scatter"]),
                "scatter": preprocess_scatter_text_for_tvs_revert(x["scatter_processed"]),
            },
            remove_columns=dataset.column_names,
        )
        return dataset

    @staticmethod
    def load_train_dataset_for_tvs_seq() -> Dataset:
        dataset = GSM8kDataset.load_train_dataset()

        dataset = dataset.map(
            lambda x: {
                "question": x["question"],
                "short_answer": x["short_answer"],
                "summarize": get_summary(x["scatter_processed"]),
                "scatter": preprocess_scatter_text_for_tvs_seq(x["scatter_processed"]),
            },
            remove_columns=dataset.column_names,
        )
        return dataset


class MHQADataset:
    dataset_name = "2wikiMultiHopQA"
    dataset_fname = "data/train_2wikimultihopqa.jsonl"
    dataset_repo = "yhytoto12/tvs-2wikimultihopqa"

    @staticmethod
    def load_train_dataset() -> Dataset:
        if os.path.exists(MHQADataset.dataset_fname):
            dataset = load_dataset("json", data_files={"train": MHQADataset.dataset_fname})["train"]
        else:
            dataset = load_dataset(MHQADataset.dataset_repo, split="train")
        return dataset

    @staticmethod
    def load_train_dataset_for_tvs_revert() -> Dataset:
        dataset = MHQADataset.load_train_dataset()

        dataset = dataset.map(
            lambda x: {
                "question": x["question"],
                "short_answer": x["short_answer"],
                "summarize": get_summary(x["scatter"]),
                "scatter": preprocess_scatter_text_for_tvs_revert(x["scatter_processed"]),
            },
            remove_columns=dataset.column_names,
        )
        return dataset

    @staticmethod
    def load_train_dataset_for_tvs_seq() -> Dataset:
        dataset = MHQADataset.load_train_dataset()

        dataset = dataset.map(
            lambda x: {
                "question": x["question"],
                "short_answer": x["short_answer"],
                "summarize": get_summary(x["scatter_processed"]),
                "scatter": preprocess_scatter_text_for_tvs_seq(x["scatter_processed"]),
            },
            remove_columns=dataset.column_names,
        )
        return dataset