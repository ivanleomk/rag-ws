from datasets import load_dataset, Dataset
from tqdm import tqdm
import hashlib
import json
from lib.models import ArxivPaper
from pydantic import BaseModel

remain_count = {}


def get_dataset(n: int):
    return load_dataset("ms_marco", "v1.1", split="train", streaming=True).take(n)


def is_valid_category(
    item, desired_categories: list[str], remain_count: dict[str, int]
):
    if (
        item["categories"][0] in desired_categories
        and remain_count[item["categories"][0]] > 0
    ):
        remain_count[item["categories"][0]] -= 1
        return True
    return False


def download_arxiv_dataset(desired_category_count: int):
    desired_categories = ["cs.AI", "cs.IR", "stat.ML"]
    remain_count = {category: desired_category_count for category in desired_categories}
    return (
        load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
        .filter(
            lambda example, idx: is_valid_category(
                example, desired_categories, remain_count
            ),
            with_indices=True,
        )
        .take(desired_category_count * len(desired_categories))
    )


def format_arxiv_dataset(ds: Dataset) -> list[ArxivPaper]:
    data = []
    for row in tqdm(ds, desc="Formatting entries"):
        combined_chunk = f"Title:{row['title']}\nAbstract:{row['abstract']}"
        data.append(
            ArxivPaper(
                **{
                    "title": row["title"],
                    "authors": row["authors"],
                    "category": row["categories"][0],
                    "abstract": row["abstract"],
                    "text": combined_chunk,
                    "chunk_id": hashlib.md5(combined_chunk.encode()).hexdigest(),
                }
            )
        )

    return data


def generate_data_and_labels(dataset: Dataset):
    passages = set()
    data = []
    labels = []
    for row in tqdm(dataset):
        selected_passages = []
        for idx, passage in enumerate(row["passages"]["passage_text"]):
            if passage not in passages:
                chunk_id = hashlib.md5(passage.encode()).hexdigest()
                passage_data_obj = {"text": passage, "chunk_id": chunk_id}
                data.append(passage_data_obj)
                passages.add(passage)

                if row["passages"]["is_selected"][idx]:
                    selected_passages.append(passage_data_obj)

        if selected_passages:
            labels.append(
                {
                    "query": row["query"],
                    "selected_passages": selected_passages,
                    "answer": row["answers"],
                    "query_id": row["query_id"],
                    "query_type": row["query_type"],
                }
            )

    print(f"Extracted {len(passages)} unique passages and {len(labels)} test queries")
    return data, labels


def save_labels(labels: list[object | BaseModel], file_path):
    with open(file_path, "w") as file:
        for label in labels:
            json_string = (
                label.model_dump_json()
                if isinstance(label, BaseModel)
                else json.dumps(label)
            )
            file.write(json_string + "\n")
    print(f"Labels saved to {file_path}")


def generate_category_test_labels(labels):
    res = []
    for row in labels:
        res.append(
            {
                "query": row["response"].question,
                "category": row["source"].category,
                "chunk_id": row["source"].chunk_id,
            }
        )
    return res


def generate_test_labels(labels, has_single_label: bool):
    res = []

    for row in labels:
        if has_single_label:
            for passage in row["selected_passages"]:
                res.append(
                    {
                        "query": row["query"],
                        "selected_chunk_ids": passage["chunk_id"],
                    }
                )
        else:
            res.append(
                {
                    "query": row["query"],
                    "selected_chunk_ids": [
                        passage["chunk_id"] for passage in row["selected_passages"]
                    ],
                }
            )

    return res


def get_labels(file_path):
    """
    We assume that this is a .jsonl file
    """
    with open(file_path, "r") as f:
        labels = []
        for line in f:
            labels.append(json.loads(line))
    return labels
