from datasets import load_dataset, Dataset
import json
import sys
import os
from typing import List, Dict, Any
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_eval_dataset(dataset_name: str, num_samples: int = None) -> Dataset:
    if dataset_name == "gsm8k":
        dataset = GSM8kEvalDataset.load_eval_dataset()
    elif dataset_name == "scibench":
        dataset = SciBenchEvalDataset.load_eval_dataset()
    elif dataset_name == "2wikimultihopqa":
        dataset = MHQAEvalDataset.load_eval_dataset()
    elif dataset_name.endswith(".json"):
        with open(dataset_name, "r") as f:
            dataset = json.load(f)

        dataset = Dataset.from_list(dataset["results"])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if num_samples is not None:
        if num_samples > len(dataset):
            print(
                f"[Warning] num_samples ({num_samples}) is greater than the dataset size ({len(dataset)}). "
                "Using the entire dataset instead."
            )
        else:
            print(f"Using {num_samples} / {len(dataset)} samples from the dataset.")
            dataset = dataset.select(range(num_samples))
    else:
        print("Total number of samples in the dataset:", len(dataset))

    return dataset

def load_eval_dataset_for_training(dataset_names: List[str], think_model_name_or_path: str, num_samples_per_dataset: int = 2):
    datasets = {}

    for dataset_name in dataset_names:

        dataset_path = os.path.join("results", f"eval_{dataset_name}_{think_model_name_or_path.split('/')[-1]}.json")
        dataset = load_eval_dataset(dataset_path, num_samples_per_dataset)

        datasets[dataset_name] = dataset

    return datasets



class GSM8kEvalDataset(Dataset):
    dataset_name = "gsm8k"

    @staticmethod
    def load_eval_dataset() -> Dataset:
        dataset = load_dataset("openai/gsm8k", "main")["test"]

        dataset = dataset.map(
            lambda x: {"question": x["question"], "short_answer": x["answer"].split("####")[-1].strip()},
            remove_columns=dataset.column_names,
        )

        return dataset


class SciBenchEvalDataset(Dataset):
    dataset_name = "scibench"

    @staticmethod
    def load_eval_dataset() -> Dataset:
        dataset = load_dataset("xw27/scibench")["train"]

        def process(x):
            question = x["problem_text"]
            answer = x["answer_number"]
            unit = x["unit"]
            if unit.strip() != "":
                question = question + f" Answer in {unit}."

            return {
                "question": question,
                "short_answer": answer,
                "unit": unit,
            }

        dataset = dataset.map(
            process,
            remove_columns=dataset.column_names,
        )

        return dataset


class MHQAEvalDataset(Dataset):
    dataset_name = "2wikimultihopqa"

    @staticmethod
    def load_eval_dataset() -> Dataset:
        dataset = load_dataset("json", data_files="data/2wikiMHQA_dev_subsample.jsonl")["train"]

        return dataset
